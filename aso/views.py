import csv
import json
import logging
import re
import threading
import time
import urllib.request
from datetime import timedelta

from django.conf import settings
from django.db import close_old_connections
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .forms import AppForm, KeywordSearchForm, OpportunitySearchForm, RuntimeConfigForm, COUNTRY_CHOICES
from .models import (
    ASCMetricDaily,
    ASCSyncRun,
    AICopilotMetadataVariant,
    AICopilotRecommendation,
    AICopilotRun,
    AISuggestion,
    AISuggestionRun,
    App,
    Keyword,
    RefreshRun,
    RuntimeConfig,
    SearchResult,
)
from .ai_service import (
    AISuggestionError,
    accept_ai_suggestion,
    generate_ai_suggestions,
    reject_ai_suggestion,
)
from .app_store_connect_service import ASCAPIError, ASCAuthError, ASCError, AppStoreConnectService
from .copilot_ai_service import (
    AICopilotError,
    CopilotSettings,
    accept_copilot_metadata_variant,
    accept_copilot_recommendation,
    generate_ai_copilot,
    reject_copilot_metadata_variant,
    reject_copilot_recommendation,
)
from .services import (
    DifficultyCalculator,
    DownloadEstimator,
    ITunesSearchService,
    PopularityEstimator,
)
from .refresh_service import run_refresh

logger = logging.getLogger(__name__)
BULK_REFRESH_STALE_HOURS = 4
COPILOT_STALE_MINUTES = 45


def _run_bulk_refresh_in_background(*, run_id: int, pairs: list[tuple[int, str]]) -> None:
    close_old_connections()
    try:
        run_refresh(
            trigger=RefreshRun.TRIGGER_MANUAL,
            source=SearchResult.SOURCE_BULK_REFRESH,
            force=True,
            pairs_override=pairs,
            run_id=run_id,
        )
    except Exception as exc:  # pragma: no cover - defensive production path
        logger.exception("Background bulk refresh failed for run %s", run_id)
        RefreshRun.objects.filter(pk=run_id).update(
            status=RefreshRun.STATUS_FAILED,
            finished_at=timezone.now(),
            current_keyword="",
            error=str(exc),
        )
    finally:
        close_old_connections()


def _mark_copilot_run_failed(*, run_id: int, message: str) -> None:
    current = AICopilotRun.objects.filter(pk=run_id).values_list("progress_percent", flat=True).first() or 0
    AICopilotRun.objects.filter(pk=run_id).update(
        status=AICopilotRun.STATUS_FAILED,
        finished_at=timezone.now(),
        error=str(message or "Copilot run failed."),
        progress_percent=max(5, int(current)),
        progress_stage="failed",
        progress_detail=str(message or "Copilot run failed.")[:255],
    )


def _run_copilot_in_background(
    *,
    run_id: int,
    app_id: int,
    country: str,
    model: str,
    sync_asc: bool,
    effective_settings: dict,
) -> None:
    close_old_connections()
    try:
        app = App.objects.get(pk=app_id)
        run = AICopilotRun.objects.get(pk=run_id)

        if sync_asc:
            AICopilotRun.objects.filter(pk=run_id, status=AICopilotRun.STATUS_RUNNING).update(
                progress_percent=8,
                progress_stage="syncing_asc",
                progress_detail="Syncing App Store Connect analytics before generation.",
            )
            resolved_asc_app_id = _effective_asc_app_id(app)
            if not resolved_asc_app_id:
                raise AICopilotError("This app is missing asc_app_id and has no track_id fallback.")

            asc_run = ASCSyncRun.objects.create(
                app=app,
                status=ASCSyncRun.STATUS_RUNNING,
                days_back=effective_settings["asc_default_days_back"],
                rows_upserted=0,
            )
            try:
                asc_service = _build_asc_service(effective_settings)
                rows_upserted = asc_service.sync_app_metrics(
                    app,
                    days_back=effective_settings["asc_default_days_back"],
                    asc_app_id=resolved_asc_app_id,
                )
                asc_run.status = ASCSyncRun.STATUS_SUCCESS
                asc_run.rows_upserted = rows_upserted
                asc_run.finished_at = timezone.now()
                asc_run.error = ""
                asc_run.save(update_fields=["status", "rows_upserted", "finished_at", "error"])
            except (ASCAuthError, ASCError, ASCAPIError) as exc:
                logger.warning(
                    "ASC pre-sync failed during background copilot run for app=%s asc_app_id=%s: %s",
                    app.id,
                    resolved_asc_app_id,
                    exc,
                )
                asc_run.status = ASCSyncRun.STATUS_FAILED
                asc_run.rows_upserted = 0
                asc_run.finished_at = timezone.now()
                asc_run.error = str(exc)
                asc_run.save(update_fields=["status", "rows_upserted", "finished_at", "error"])
                AICopilotRun.objects.filter(pk=run_id, status=AICopilotRun.STATUS_RUNNING).update(
                    progress_percent=12,
                    progress_stage="asc_sync_failed",
                    progress_detail=f"ASC pre-sync failed ({exc}); continuing with existing data."[:255],
                )
            except Exception as exc:
                logger.exception(
                    "ASC pre-sync crashed during background copilot run for app=%s asc_app_id=%s",
                    app.id,
                    resolved_asc_app_id,
                )
                asc_run.status = ASCSyncRun.STATUS_FAILED
                asc_run.rows_upserted = 0
                asc_run.finished_at = timezone.now()
                asc_run.error = str(exc)
                asc_run.save(update_fields=["status", "rows_upserted", "finished_at", "error"])
                AICopilotRun.objects.filter(pk=run_id, status=AICopilotRun.STATUS_RUNNING).update(
                    progress_percent=12,
                    progress_stage="asc_sync_failed",
                    progress_detail="ASC pre-sync failed unexpectedly; continuing with existing data.",
                )

        run.refresh_from_db()
        generate_ai_copilot(
            app,
            country=country,
            settings_obj=CopilotSettings(
                model=model,
                api_key=effective_settings["api_key"],
                system_prompt=effective_settings["system_prompt"],
                user_prompt_template=effective_settings["user_prompt_template"],
            ),
            run=run,
        )
    except AICopilotError as exc:
        logger.warning("AI copilot background run %s failed for app %s: %s", run_id, app_id, exc)
        _mark_copilot_run_failed(run_id=run_id, message=str(exc))
    except Exception as exc:  # pragma: no cover - defensive runtime path
        logger.exception("AI copilot background run %s crashed for app %s", run_id, app_id)
        _mark_copilot_run_failed(run_id=run_id, message=str(exc))
    finally:
        close_old_connections()


def _start_copilot_background_run(
    *,
    run_id: int,
    app_id: int,
    country: str,
    model: str,
    sync_asc: bool,
    effective_settings: dict,
) -> None:
    thread = threading.Thread(
        target=_run_copilot_in_background,
        kwargs={
            "run_id": run_id,
            "app_id": app_id,
            "country": country,
            "model": model,
            "sync_asc": sync_asc,
            "effective_settings": effective_settings,
        },
        daemon=True,
        name=f"copilot-run-{run_id}",
    )
    thread.start()


# app_rank is now persisted directly on SearchResult during search/refresh.
# No need for a helper to find rank in stored competitors.


def methodology_view(request):
    """Our Methodology page — explains how RespectASO works."""
    return render(request, "aso/methodology.html")


def setup_view(request):
    """Setup guide — custom domain, Docker config, and getting started."""
    return render(request, "aso/setup.html")


def config_view(request):
    """In-app runtime configuration (OpenAI + App Store Connect overrides)."""
    runtime_config = RuntimeConfig.get_solo()
    saved = False

    if request.method == "POST":
        form = RuntimeConfigForm(request.POST, config=runtime_config)
        if form.is_valid():
            runtime_config = form.save()
            saved = True
    else:
        form = RuntimeConfigForm(config=runtime_config)

    effective = _effective_ai_settings()
    return render(
        request,
        "aso/config.html",
        {
            "form": form,
            "saved": saved,
            "api_key_set": bool(effective["api_key"]),
            "api_key_source": effective["api_key_source"],
            "effective_default_model": effective["default_model"],
            "effective_available_models": effective["available_models"],
            "effective_online_context_enabled": effective["enable_online_context"],
            "effective_online_top_apps_per_country": effective["online_top_apps_per_country"],
            "effective_history_rows_max": effective["history_rows_max"],
            "asc_configured": effective["asc_configured"],
            "asc_key_source": effective["asc_key_source"],
            "effective_asc_default_days_back": effective["asc_default_days_back"],
        },
    )


def _openai_test_error_payload(exc: Exception) -> tuple[dict, int]:
    status_code = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    response = getattr(exc, "response", None)
    response_text = getattr(response, "text", None) if response is not None else None
    body_obj = getattr(exc, "body", None)
    body_excerpt = ""
    if response_text:
        body_excerpt = str(response_text).replace("\n", " ").strip()[:320]
    elif body_obj:
        if isinstance(body_obj, (dict, list)):
            body_excerpt = json.dumps(body_obj, ensure_ascii=True)[:320]
        else:
            body_excerpt = str(body_obj).replace("\n", " ").strip()[:320]

    message = str(exc or "").strip() or exc.__class__.__name__
    lower = f"{message} {body_excerpt}".lower()

    def _is_openai_exc(name: str) -> bool:
        try:
            import openai
        except Exception:  # pragma: no cover - import guard
            return False
        exc_type = getattr(openai, name, None)
        return isinstance(exc_type, type) and isinstance(exc, exc_type)

    payload = {
        "success": False,
        "error": f"OpenAI request failed: {message}",
        "error_kind": "openai_error",
        "status_code": status_code,
        "request_id": request_id,
        "error_type": exc.__class__.__name__,
        "detail": body_excerpt or None,
    }
    http_status = 502

    if _is_openai_exc("APITimeoutError") or "timed out" in lower or "timeout" in lower:
        payload["error"] = (
            "OpenAI request timed out. Retry, or increase OPENAI_TIMEOUT_SECONDS in Config/env."
        )
        payload["error_kind"] = "timeout"
        http_status = 504
    elif _is_openai_exc("APIConnectionError"):
        payload["error"] = "Could not connect to OpenAI. Check network egress and retry."
        payload["error_kind"] = "connection"
        http_status = 503
    elif _is_openai_exc("RateLimitError") or status_code == 429 or "insufficient_quota" in lower or "quota" in lower:
        payload["error"] = "OpenAI quota or rate limit was hit. Check billing/usage and retry."
        payload["error_kind"] = "quota_or_rate_limit"
        http_status = 429
    elif _is_openai_exc("AuthenticationError") or status_code == 401 or "invalid api key" in lower:
        payload["error"] = "OpenAI authentication failed. Verify OPENAI_API_KEY."
        payload["error_kind"] = "authentication"
        http_status = 400
    elif _is_openai_exc("PermissionDeniedError") or status_code == 403:
        payload["error"] = "OpenAI request was forbidden for this key/model. Check project permissions."
        payload["error_kind"] = "permission"
        http_status = 400
    elif _is_openai_exc("BadRequestError") or status_code == 400:
        payload["error"] = f"OpenAI rejected the request: {message}"
        payload["error_kind"] = "bad_request"
        http_status = 400
    elif status_code and int(status_code) >= 500:
        payload["error"] = "OpenAI service is temporarily unavailable. Please retry."
        payload["error_kind"] = "upstream_unavailable"
        http_status = 503

    return payload, http_status


@require_POST
def config_openai_test_view(request):
    try:
        body = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    effective = _effective_ai_settings()
    api_key_override = str(body.get("api_key") or "").strip()
    api_key = api_key_override or effective["api_key"]
    if not api_key:
        return JsonResponse(
            {
                "success": False,
                "error": "OpenAI API key is not configured. Add one in Config or OPENAI_API_KEY.",
                "error_kind": "missing_api_key",
            },
            status=400,
        )

    allowed_models_override = _parse_csv_models(body.get("available_models"))
    allowed_models = allowed_models_override or effective["available_models"]
    try:
        selected_model = _resolve_model_choice(
            body.get("model"),
            default_model=effective["default_model"],
            allowed_models=allowed_models,
        )
    except AISuggestionError as exc:
        return JsonResponse({"success": False, "error": str(exc), "error_kind": "invalid_model"}, status=400)

    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - import guard
        return JsonResponse(
            {
                "success": False,
                "error": "OpenAI SDK is not installed. Add `openai` to requirements.",
                "error_kind": "sdk_missing",
                "detail": str(exc),
            },
            status=500,
        )

    started = time.perf_counter()
    try:
        client = OpenAI(
            api_key=api_key,
            timeout=settings.OPENAI_TIMEOUT_SECONDS,
            max_retries=settings.OPENAI_MAX_RETRIES,
        )
        response = client.with_options(
            timeout=min(float(settings.OPENAI_TIMEOUT_SECONDS), 30.0),
            max_retries=settings.OPENAI_MAX_RETRIES,
        ).responses.create(
            model=selected_model,
            input="Reply with a short confirmation that this model is reachable.",
            max_output_tokens=32,
        )
        elapsed = round(time.perf_counter() - started, 2)
        output_text = (getattr(response, "output_text", "") or "").strip()
        request_id = getattr(response, "_request_id", None)
        logger.warning(
            "OpenAI config test succeeded model=%s request_id=%s elapsed=%.2fs output=%s",
            selected_model,
            request_id or "-",
            elapsed,
            output_text[:120] or "-",
        )
        return JsonResponse(
            {
                "success": True,
                "model": selected_model,
                "elapsed_seconds": elapsed,
                "request_id": request_id,
                "output_preview": output_text[:240],
            }
        )
    except Exception as exc:
        payload, http_status = _openai_test_error_payload(exc)
        payload["model"] = selected_model
        payload["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        logger.warning(
            "OpenAI config test failed model=%s kind=%s status=%s request_id=%s detail=%s",
            selected_model,
            payload.get("error_kind"),
            payload.get("status_code"),
            payload.get("request_id") or "-",
            exc,
        )
        return JsonResponse(payload, status=http_status)


def dashboard_view(request):
    """
    Main dashboard with keyword search bar, results, and full search history.

    Shows only the latest result per keyword+country pair.  Each result
    is annotated with trend data (comparison to previous result) for
    inline ↑↓ indicators.
    """
    apps = App.objects.all()
    search_form = KeywordSearchForm()

    # --- History table (latest result per keyword+country) ---
    app_id = request.GET.get("app")
    country_filter = request.GET.get("country", "")
    sort_key = str(request.GET.get("sort", "date")).strip().lower()
    sort_dir = str(request.GET.get("dir", "desc")).strip().lower()
    valid_sort_keys = {
        "keyword",
        "rank",
        "popularity",
        "difficulty",
        "country",
        "competitors",
        "date",
    }
    if sort_key not in valid_sort_keys:
        sort_key = "date"
    if sort_dir not in {"asc", "desc"}:
        sort_dir = "desc"

    # Get the latest result ID for each keyword+country pair
    from django.db.models import Max

    latest_filter = {}
    if app_id:
        latest_filter["keyword__app_id"] = app_id
    if country_filter:
        latest_filter["country"] = country_filter.lower()

    latest_ids_qs = (
        SearchResult.objects
        .filter(**latest_filter)
        .values("keyword_id", "country")
        .annotate(latest_id=Max("id"))
        .values_list("latest_id", flat=True)
    )

    # Distinct countries that have results (for the history country filter)
    country_base_filter = {}
    if app_id:
        country_base_filter["keyword__app_id"] = app_id
    available_countries = (
        SearchResult.objects
        .filter(**country_base_filter)
        .values_list("country", flat=True)
        .distinct()
        .order_by("country")
    )
    latest_ids = list(latest_ids_qs)

    results_qs = (
        SearchResult.objects
        .filter(id__in=latest_ids)
        .select_related("keyword", "keyword__app")
    )
    sorted_results = list(results_qs)

    def _sort_results(rows):
        if sort_key == "keyword":
            rows.sort(key=lambda r: (r.keyword.keyword or "").lower(), reverse=(sort_dir == "desc"))
        elif sort_key == "rank":
            if sort_dir == "asc":
                rows.sort(key=lambda r: (r.app_rank is None, r.app_rank if r.app_rank is not None else 10**9))
            else:
                rows.sort(key=lambda r: (r.app_rank is None, -(r.app_rank or 0)))
        elif sort_key == "popularity":
            if sort_dir == "asc":
                rows.sort(
                    key=lambda r: (
                        r.popularity_score is None,
                        r.popularity_score if r.popularity_score is not None else 10**9,
                    )
                )
            else:
                rows.sort(
                    key=lambda r: (
                        r.popularity_score is None,
                        -(r.popularity_score or 0),
                    )
                )
        elif sort_key == "difficulty":
            rows.sort(key=lambda r: r.difficulty_score or 0, reverse=(sort_dir == "desc"))
        elif sort_key == "country":
            rows.sort(key=lambda r: (r.country or "").lower(), reverse=(sort_dir == "desc"))
        elif sort_key == "competitors":
            rows.sort(key=lambda r: len(r.competitors_data or []), reverse=(sort_dir == "desc"))
        else:  # date
            rows.sort(key=lambda r: r.searched_at or timezone.now(), reverse=(sort_dir == "desc"))

    _sort_results(sorted_results)

    # Count unique keywords for the toolbar
    keyword_qs = Keyword.objects.all()
    if app_id:
        keyword_qs = keyword_qs.filter(app_id=app_id)
    keyword_count = keyword_qs.count()

    # Pagination (25 per page)
    page = request.GET.get("page", "1")
    try:
        page = max(1, int(page))
    except (ValueError, TypeError):
        page = 1

    per_page = 25
    total_count = len(sorted_results)
    total_pages = max(1, (total_count + per_page - 1) // per_page)
    page = min(page, total_pages)
    start = (page - 1) * per_page
    history_results = sorted_results[start : start + per_page]

    # Annotate each result with trend data (previous result comparison)
    for result in history_results:
        prev = (
            SearchResult.objects
            .filter(
                keyword_id=result.keyword_id,
                country=result.country,
                searched_at__lt=result.searched_at,
            )
            .order_by("-searched_at")
            .first()
        )
        history_count = SearchResult.objects.filter(
            keyword_id=result.keyword_id, country=result.country
        ).count()
        result.has_history = history_count > 1
        if prev:
            result.prev_popularity = prev.popularity_score
            result.prev_difficulty = prev.difficulty_score
            result.prev_rank = prev.app_rank
            # Calculate deltas
            if result.popularity_score is not None and prev.popularity_score is not None:
                result.popularity_delta = result.popularity_score - prev.popularity_score
            else:
                result.popularity_delta = None
            result.difficulty_delta = result.difficulty_score - prev.difficulty_score
            if result.app_rank is not None and prev.app_rank is not None:
                result.rank_delta = prev.app_rank - result.app_rank  # Lower rank = better = positive delta
            else:
                result.rank_delta = None
        else:
            result.prev_popularity = None
            result.prev_difficulty = None
            result.prev_rank = None
            result.popularity_delta = None
            result.difficulty_delta = None
            result.rank_delta = None

    # Show rank column whenever rank can be meaningful in current scope.
    show_rank = False
    selected_app_name = None
    has_rank_data = any(result.app_rank is not None for result in sorted_results)
    has_trackable_rows = any(
        bool(result.keyword.app and result.keyword.app.track_id)
        for result in sorted_results
    )
    if app_id:
        selected_app_obj = App.objects.filter(id=app_id).first()
        if selected_app_obj:
            selected_app_name = selected_app_obj.name
            show_rank = bool(selected_app_obj.track_id or has_rank_data)
    else:
        show_rank = bool(has_trackable_rows or has_rank_data)

    base_params = request.GET.copy()
    if "page" in base_params:
        base_params.pop("page")
    sort_links = {}
    for key in valid_sort_keys:
        params = base_params.copy()
        next_dir = "desc" if (sort_key == key and sort_dir == "asc") else "asc"
        params["sort"] = key
        params["dir"] = next_dir
        sort_links[key] = params.urlencode()

    prev_query = ""
    next_query = ""
    if page > 1:
        prev_params = base_params.copy()
        prev_params["page"] = page - 1
        prev_query = prev_params.urlencode()
    if page < total_pages:
        next_params = base_params.copy()
        next_params["page"] = page + 1
        next_query = next_params.urlencode()

    return render(
        request,
        "aso/dashboard.html",
        {
            "apps": apps,
            "search_form": search_form,
            # History table context
            "history_results": history_results,
            "keyword_count": keyword_count,
            "selected_app": int(app_id) if app_id else None,
            "selected_app_name": selected_app_name,
            "selected_country": country_filter,
            "available_countries": list(available_countries),
            "show_rank": show_rank,
            "page": page,
            "total_pages": total_pages,
            "total_count": total_count,
            "has_prev": page > 1,
            "has_next": page < total_pages,
            "current_sort": sort_key,
            "current_sort_dir": sort_dir,
            "sort_links": sort_links,
            "prev_query": prev_query,
            "next_query": next_query,
        },
    )


@require_POST
def search_view(request):
    """
    Process keyword search request across one or more countries (max 5).

    Accepts comma-separated keywords (max 20) and comma-separated countries (max 5).
    For each keyword × country combination:
      1. Search iTunes for top competitors
      2. Calculate difficulty score
      3. Estimate popularity from competitor data
      4. Save results to DB
    Returns JSON with results grouped by country.
    """
    form = KeywordSearchForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"error": "Invalid form data."}, status=400)

    raw_keywords = form.cleaned_data["keywords"]
    app_id = form.cleaned_data.get("app_id")
    countries = form.cleaned_data.get("countries", ["us"])

    # Parse comma-separated keywords, limit to 20
    keywords = [kw.strip() for kw in raw_keywords.split(",") if kw.strip()][:20]

    if not keywords:
        return JsonResponse({"error": "No keywords provided."}, status=400)

    # Get app if specified
    app = None
    if app_id:
        try:
            app = App.objects.get(id=app_id)
        except App.DoesNotExist:
            pass

    # Set up services
    itunes_service = ITunesSearchService()
    difficulty_calc = DifficultyCalculator()
    popularity_est = PopularityEstimator()
    download_est = DownloadEstimator()

    # Results grouped by country
    results_by_country = {}
    skipped = []
    call_count = 0

    for country in countries:
        country_results = []
        for kw_text in keywords:
            # Rate limit between API calls
            if call_count > 0:
                time.sleep(2)
            call_count += 1

            # Get or create keyword
            keyword_obj, created = Keyword.objects.get_or_create(
                keyword=kw_text.lower(),
                app=app,
            )

            # Skip if this keyword already has results for the SAME country today
            today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
            if not created and keyword_obj.results.filter(
                country=country, searched_at__gte=today_start
            ).exists():
                skipped.append(f"{kw_text} ({country.upper()})")
                continue

            # iTunes Search
            competitors = itunes_service.search_apps(kw_text, country=country, limit=25)

            # Difficulty Score
            difficulty_score, breakdown = difficulty_calc.calculate(
                competitors, keyword=kw_text
            )

            # Find user's app rank (if app has a track_id)
            app_rank = None
            if app and app.track_id:
                app_rank = itunes_service.find_app_rank(
                    kw_text, app.track_id, country=country
                )

            # Popularity (estimated from competitor data)
            popularity = popularity_est.estimate(competitors, kw_text)

            # Download estimates
            download_estimates = download_est.estimate(popularity or 0, len(competitors))
            breakdown["download_estimates"] = download_estimates

            # Save result (one entry per keyword+country per day)
            search_result = SearchResult.upsert_today(
                keyword=keyword_obj,
                popularity_score=popularity,
                difficulty_score=difficulty_score,
                difficulty_breakdown=breakdown,
                competitors_data=competitors,
                app_rank=app_rank,
                country=country,
            )

            country_results.append(
                {
                    "keyword": kw_text,
                    "country": country,
                    "popularity_score": popularity,
                    "difficulty_score": difficulty_score,
                    "difficulty_label": search_result.difficulty_label,
                    "difficulty_color": search_result.difficulty_color,
                    "difficulty_breakdown": breakdown,
                    "competitors": competitors,
                    "result_id": search_result.id,
                    "app_rank": app_rank,
                    "app_name": app.name if app else None,
                    "app_icon": app.icon_url if app else None,
                }
            )
        results_by_country[country] = country_results

    # Build opportunity ranking when multiple countries searched
    opportunity_ranking = []
    if len(countries) > 1:
        # Group results by keyword across countries
        kw_map = {}
        for country, cresults in results_by_country.items():
            for r in cresults:
                kw = r["keyword"]
                if kw not in kw_map:
                    kw_map[kw] = {}
                pop = r["popularity_score"] or 0
                diff = r["difficulty_score"]
                opp = round(pop * (100 - diff) / 100)
                kw_map[kw][country] = {
                    "popularity": pop,
                    "difficulty": diff,
                    "opportunity": opp,
                }
        for kw, country_data in kw_map.items():
            best_country = max(country_data, key=lambda c: country_data[c]["opportunity"])
            opportunity_ranking.append({
                "keyword": kw,
                "countries": country_data,
                "best_country": best_country,
                "best_score": country_data[best_country]["opportunity"],
            })
        opportunity_ranking.sort(key=lambda x: x["best_score"], reverse=True)

    response_data = {
        "results_by_country": results_by_country,
        "countries": countries,
        "opportunity_ranking": opportunity_ranking,
    }
    if skipped:
        response_data["skipped"] = skipped
        response_data["warning"] = (
            f"Skipped {len(skipped)} keyword(s) already in your list: "
            + ", ".join(skipped)
            + ". Use Refresh to update them."
        )
    return JsonResponse(response_data)


def opportunity_view(request):
    """Country Opportunity Finder — search a keyword across all 30 countries."""
    apps = App.objects.all()
    form = OpportunitySearchForm()
    return render(request, "aso/opportunity.html", {"apps": apps, "form": form})


@require_POST
def opportunity_search_view(request):
    """
    AJAX endpoint: search a single keyword across all 30 countries.

    Returns ranked list of countries by opportunity score.
    """
    form = OpportunitySearchForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"error": "Invalid form data."}, status=400)

    kw_text = form.cleaned_data["keyword"].strip().lower()
    app_id = form.cleaned_data.get("app_id")

    if not kw_text:
        return JsonResponse({"error": "No keyword provided."}, status=400)

    app = None
    if app_id:
        try:
            app = App.objects.get(id=app_id)
        except App.DoesNotExist:
            pass

    itunes_service = ITunesSearchService()
    difficulty_calc = DifficultyCalculator()
    popularity_est = PopularityEstimator()
    download_est = DownloadEstimator()

    results = []
    for i, (country_code, country_name) in enumerate(COUNTRY_CHOICES):
        if i > 0:
            time.sleep(2)

        competitors = itunes_service.search_apps(kw_text, country=country_code, limit=25)
        difficulty_score, breakdown = difficulty_calc.calculate(
            competitors, keyword=kw_text
        )
        popularity = popularity_est.estimate(competitors, kw_text)

        download_estimates = download_est.estimate(popularity or 0, len(competitors))
        breakdown["download_estimates"] = download_estimates

        app_rank = None
        if app and app.track_id:
            app_rank = itunes_service.find_app_rank(
                kw_text, app.track_id, country=country_code
            )

        # Compute difficulty label from score (same logic as model property)
        if difficulty_score <= 15:
            diff_label = "Very Easy"
        elif difficulty_score <= 35:
            diff_label = "Easy"
        elif difficulty_score <= 55:
            diff_label = "Moderate"
        elif difficulty_score <= 75:
            diff_label = "Hard"
        elif difficulty_score <= 90:
            diff_label = "Very Hard"
        else:
            diff_label = "Extreme"

        opportunity = round(popularity * (100 - difficulty_score) / 100) if popularity else 0
        top_competitor = competitors[0]["trackName"] if competitors else "—"
        top_ratings = competitors[0].get("userRatingCount", 0) if competitors else 0

        results.append({
            "country": country_code,
            "popularity": popularity,
            "difficulty": difficulty_score,
            "difficulty_label": diff_label,
            "difficulty_breakdown": breakdown,
            "competitors_data": competitors,
            "opportunity": opportunity,
            "app_rank": app_rank,
            "competitor_count": len(competitors),
            "top_competitor": top_competitor,
            "top_ratings": top_ratings,
        })

    results.sort(key=lambda x: x["opportunity"], reverse=True)

    return JsonResponse({
        "keyword": kw_text,
        "app_id": app.id if app else None,
        "results": results,
        "total_countries": len(results),
    })


@require_POST
def opportunity_save_view(request):
    """
    Save selected opportunity results to search history.

    Accepts JSON body with keyword, app_id, and selected results
    (each containing country, popularity, difficulty, breakdown, competitors, etc.).
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON."}, status=400)

    kw_text = body.get("keyword", "").strip().lower()
    app_id = body.get("app_id")
    selected = body.get("results", [])

    if not kw_text or not selected:
        return JsonResponse({"error": "No keyword or results provided."}, status=400)

    app = None
    if app_id:
        try:
            app = App.objects.get(id=app_id)
        except App.DoesNotExist:
            pass

    keyword_obj, _ = Keyword.objects.get_or_create(keyword=kw_text, app=app)
    saved = 0

    for item in selected:
        country = item.get("country", "us")
        SearchResult.create_snapshot(
            keyword=keyword_obj,
            popularity_score=item.get("popularity", 0),
            difficulty_score=item.get("difficulty", 0),
            difficulty_breakdown=item.get("difficulty_breakdown", {}),
            competitors_data=item.get("competitors_data", []),
            app_rank=item.get("app_rank"),
            country=country,
            source=SearchResult.SOURCE_OPPORTUNITY_SAVE,
        )
        saved += 1

    return JsonResponse({"success": True, "saved": saved})


def app_lookup_view(request):
    """
    AJAX endpoint: search the App Store for apps by name or URL.

    Accepts GET parameter 'q' — either:
      - An App Store URL (https://apps.apple.com/...id123456789)
      - A search query (app name)

    Returns JSON list of matching apps with icon, name, bundle_id, track_id.
    """
    query = request.GET.get("q", "").strip()
    if not query or len(query) < 2:
        return JsonResponse({"apps": []})

    itunes_service = ITunesSearchService()

    # Check if the query is an App Store URL
    url_match = re.search(r"/id(\d+)", query)
    if url_match:
        track_id = int(url_match.group(1))
        app_data = itunes_service.lookup_by_id(track_id)
        if app_data:
            return JsonResponse(
                {
                    "apps": [
                        {
                            "trackId": app_data["trackId"],
                            "trackName": app_data["trackName"],
                            "artworkUrl100": app_data["artworkUrl100"],
                            "bundleId": app_data["bundleId"],
                            "sellerName": app_data["sellerName"],
                        }
                    ]
                }
            )
        return JsonResponse({"apps": []})

    # Otherwise search by name
    results = itunes_service.search_apps(query, limit=5)
    return JsonResponse(
        {
            "apps": [
                {
                    "trackId": r["trackId"],
                    "trackName": r["trackName"],
                    "artworkUrl100": r["artworkUrl100"],
                    "bundleId": r["bundleId"],
                    "sellerName": r["sellerName"],
                }
                for r in results
            ]
        }
    )


def apps_view(request):
    """
    Manage apps for keyword categorization.

    Supports two flows:
      1. Manual entry (name + optional bundle_id)
      2. App Store lookup (sets track_id, icon, seller from iTunes data)
    """
    message = None
    message_type = None

    if request.method == "POST":
        # Check if this is from App Store lookup (has track_id)
        track_id = request.POST.get("track_id")
        if track_id:
            try:
                track_id_int = int(track_id)
                # Prevent duplicate
                if App.objects.filter(track_id=track_id_int).exists():
                    message = "This app has already been added."
                    message_type = "error"
                else:
                    App.objects.create(
                        name=request.POST.get("name", "Unknown App"),
                        bundle_id=request.POST.get("bundle_id", ""),
                        track_id=track_id_int,
                        asc_app_id=(request.POST.get("asc_app_id", "") or "").strip() or str(track_id_int),
                        store_url=request.POST.get("store_url", ""),
                        icon_url=request.POST.get("icon_url", ""),
                        seller_name=request.POST.get("seller_name", ""),
                    )
                    message = f"App '{request.POST.get('name')}' added from App Store."
                    message_type = "success"
            except (ValueError, TypeError):
                message = "Invalid app data."
                message_type = "error"
        else:
            # Manual entry
            form = AppForm(request.POST)
            if form.is_valid():
                form.save()
                message = f"App '{form.cleaned_data['name']}' created."
                message_type = "success"
            else:
                message = "Please fix the errors below."
                message_type = "error"

    form = AppForm()
    apps = App.objects.prefetch_related("keywords")

    return render(
        request,
        "aso/apps.html",
        {
            "form": form,
            "apps": apps,
            "message": message,
            "message_type": message_type,
        },
    )


@require_POST
def app_delete_view(request, app_id):
    """Delete an app. Keywords are preserved (app set to null)."""
    app = get_object_or_404(App, id=app_id)
    name = app.name
    app.delete()
    return redirect("aso:apps")


@require_POST
def keyword_delete_view(request, keyword_id):
    """Delete a keyword and all its search results."""
    keyword = get_object_or_404(Keyword, id=keyword_id)
    keyword.delete()
    return JsonResponse({"success": True})


@require_POST
def result_delete_view(request, result_id):
    """Delete a single search result. If the parent keyword has no remaining results, delete the keyword too."""
    result = get_object_or_404(SearchResult, id=result_id)
    keyword = result.keyword
    result.delete()
    # Clean up orphaned keyword (no remaining search results)
    if not keyword.results.exists():
        keyword.delete()
    return JsonResponse({"success": True})


@require_POST
def keywords_bulk_delete_view(request):
    """
    Delete all keywords for an app, or ALL keywords when no app filter is active.

    POST body: {"app_id": int|null}
    """
    body = json.loads(request.body)
    app_id = body.get("app_id")

    if app_id:
        count, _ = Keyword.objects.filter(app_id=app_id).delete()
    else:
        # No app filter → delete ALL keywords (and cascade-delete their results)
        count, _ = Keyword.objects.all().delete()

    return JsonResponse({"success": True, "deleted": count})


@require_POST
def keyword_refresh_view(request, keyword_id):
    """
    Re-run the difficulty search for a single keyword.

    Uses the keyword's existing app and the country from the request.
    Returns the new result as JSON.
    """
    keyword_obj = get_object_or_404(Keyword, id=keyword_id)
    country = request.POST.get("country", "us")

    itunes_service = ITunesSearchService()
    difficulty_calc = DifficultyCalculator()
    popularity_est = PopularityEstimator()
    download_est = DownloadEstimator()

    # Search iTunes
    competitors = itunes_service.search_apps(
        keyword_obj.keyword, country=country, limit=25
    )

    # Calculate difficulty
    difficulty_score, breakdown = difficulty_calc.calculate(
        competitors, keyword=keyword_obj.keyword
    )

    # App rank
    app_rank = None
    app = keyword_obj.app
    if app and app.track_id:
        app_rank = itunes_service.find_app_rank(
            keyword_obj.keyword, app.track_id, country=country
        )

    # Popularity (estimated from competitor data)
    popularity = popularity_est.estimate(competitors, keyword_obj.keyword)

    # Download estimates
    download_estimates = download_est.estimate(popularity or 0, len(competitors))
    breakdown["download_estimates"] = download_estimates

    search_result = SearchResult.create_snapshot(
        keyword=keyword_obj,
        popularity_score=popularity,
        difficulty_score=difficulty_score,
        difficulty_breakdown=breakdown,
        competitors_data=competitors,
        app_rank=app_rank,
        country=country,
        source=SearchResult.SOURCE_MANUAL_REFRESH,
    )

    return JsonResponse({
        "success": True,
        "result": {
            "keyword": keyword_obj.keyword,
            "keyword_id": keyword_obj.pk,
            "result_id": search_result.pk,
            "popularity_score": popularity,
            "difficulty_score": difficulty_score,
            "difficulty_label": search_result.difficulty_label,
            "difficulty_color": search_result.difficulty_color,
            "country": country,
            "searched_at": search_result.searched_at.strftime("%b %d, %H:%M"),
            "app_rank": app_rank,
            "app_name": app.name if app else None,
        },
    })


def export_history_csv_view(request):
    """
    Export all search history as a CSV file.

    Supports optional ?app= filter to limit to one app.
    """
    app_id = request.GET.get("app")
    country = request.GET.get("country")
    results_qs = SearchResult.objects.select_related("keyword", "keyword__app").order_by(
        "-searched_at"
    )
    if app_id:
        results_qs = results_qs.filter(keyword__app_id=app_id)
    if country:
        results_qs = results_qs.filter(country=country.lower())

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="respectaso-search-history.csv"'

    writer = csv.writer(response)
    writer.writerow([
        "Keyword", "App", "Country", "Popularity", "Difficulty",
        "Difficulty Label", "Rank", "Competitors", "Date",
    ])

    for r in results_qs:
        writer.writerow([
            r.keyword.keyword,
            r.keyword.app.name if r.keyword.app else "",
            r.country.upper() if r.country else "",
            r.popularity_score if r.popularity_score is not None else "",
            r.difficulty_score,
            r.difficulty_label,
            r.app_rank if r.app_rank else "",
            len(r.competitors_data) if r.competitors_data else 0,
            r.searched_at.strftime("%Y-%m-%d %H:%M") if r.searched_at else "",
        ])

    # Respectlytics attribution row
    writer.writerow([])
    writer.writerow(["Privacy-first mobile analytics — https://respectlytics.com"])

    return response


@require_POST
def keywords_bulk_refresh_view(request):
    """
    Start an async refresh for in-scope tracked keyword-country pairs.

    POST body: {"app_id": int|null}
    Returns quickly with run metadata while work continues in a background thread.
    """
    try:
        body = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid JSON body."}, status=400)

    app_id = body.get("app_id")
    if app_id in {"", None}:
        app_id = None
    else:
        try:
            app_id = int(app_id)
        except (TypeError, ValueError):
            return JsonResponse({"success": False, "error": "Invalid app_id."}, status=400)

    now = timezone.now()
    stale_cutoff = now - timedelta(hours=BULK_REFRESH_STALE_HOURS)
    latest_run = RefreshRun.objects.order_by("-started_at").first()
    if latest_run and latest_run.status == RefreshRun.STATUS_RUNNING:
        if latest_run.started_at and latest_run.started_at < stale_cutoff:
            latest_run.status = RefreshRun.STATUS_FAILED
            latest_run.finished_at = now
            latest_run.current_keyword = ""
            latest_run.error = (
                f"Marked stale after {BULK_REFRESH_STALE_HOURS}h without finishing; "
                "a new refresh was started."
            )
            latest_run.save(
                update_fields=["status", "finished_at", "current_keyword", "error"]
            )
        else:
            return JsonResponse(
                {"success": False, "error": "A refresh is already running."},
                status=409,
            )

    keywords_qs = Keyword.objects.all()
    if app_id is not None:
        keywords_qs = keywords_qs.filter(app_id=app_id)
    keyword_ids = list(keywords_qs.values_list("id", flat=True))
    if not keyword_ids:
        return JsonResponse({"success": True, "started": False, "total_pairs": 0})

    pairs = list(
        SearchResult.objects.filter(keyword_id__in=keyword_ids)
        .values_list("keyword_id", "country")
        .distinct()
        .order_by("keyword_id", "country")
    )
    seen_keyword_ids = {keyword_id for keyword_id, _ in pairs}
    pairs.extend((keyword_id, "us") for keyword_id in keyword_ids if keyword_id not in seen_keyword_ids)
    pairs = sorted((int(keyword_id), str(country or "us").lower()) for keyword_id, country in pairs)

    if not pairs:
        return JsonResponse({"success": True, "started": False, "total_pairs": 0})

    run = RefreshRun.objects.create(
        trigger=RefreshRun.TRIGGER_MANUAL,
        status=RefreshRun.STATUS_RUNNING,
        total=len(pairs),
        completed=0,
        current_keyword="",
        error="",
    )

    thread = threading.Thread(
        target=_run_bulk_refresh_in_background,
        kwargs={"run_id": run.id, "pairs": pairs},
        daemon=True,
        name=f"aso-bulk-refresh-{run.id}",
    )
    thread.start()

    return JsonResponse(
        {
            "success": True,
            "started": True,
            "run_id": run.id,
            "total_pairs": len(pairs),
        }
    )


def _serialize_ai_suggestion(s: AISuggestion):
    metrics = s.market_metrics_json or {}
    countries = metrics.get("countries", [])
    return {
        "id": s.id,
        "app_id": s.app_id,
        "run_id": s.run_id,
        "keyword": s.keyword,
        "intent": s.intent,
        "rationale": s.rationale,
        "confidence": s.confidence,
        "score_market": s.score_market,
        "score_history": s.score_history,
        "score_overall": s.score_overall,
        "status": s.status,
        "created_keyword_id": s.created_keyword_id,
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "resolved_at": s.resolved_at.isoformat() if s.resolved_at else None,
        "countries": [
            {
                "country": item.get("country"),
                "popularity": item.get("popularity"),
                "difficulty": item.get("difficulty"),
                "opportunity": item.get("opportunity"),
                "app_rank": item.get("app_rank"),
            }
            for item in countries
        ],
    }


def _ai_payload_for_app(app: App):
    runs = list(
        AISuggestionRun.objects.filter(app=app)
        .order_by("-started_at")[:20]
        .values("id", "status", "model", "countries_json", "started_at", "finished_at", "error")
    )
    suggestions_qs = AISuggestion.objects.filter(app=app).select_related("created_keyword", "run")
    suggestions_qs = suggestions_qs.order_by("-created_at")[:200]
    suggestions = [_serialize_ai_suggestion(s) for s in suggestions_qs]
    return {
        "app": {
            "id": app.id,
            "name": app.name,
            "track_id": app.track_id,
        },
        "runs": [
            {
                "id": run["id"],
                "status": run["status"],
                "model": run["model"],
                "countries": run["countries_json"],
                "started_at": run["started_at"].isoformat() if run["started_at"] else None,
                "finished_at": run["finished_at"].isoformat() if run["finished_at"] else None,
                "error": run["error"],
            }
            for run in runs
        ],
        "suggestions": suggestions,
    }


def _serialize_copilot_recommendation(rec: AICopilotRecommendation):
    evidence = rec.evidence_json or {}
    return {
        "id": rec.id,
        "run_id": rec.run_id,
        "app_id": rec.app_id,
        "keyword": rec.keyword,
        "action": rec.action,
        "rationale": rec.rationale,
        "llm_confidence": rec.llm_confidence,
        "score_market": rec.score_market,
        "score_rank_momentum": rec.score_rank_momentum,
        "score_business_impact": rec.score_business_impact,
        "score_coverage_gap": rec.score_coverage_gap,
        "score_overall": rec.score_overall,
        "status": rec.status,
        "created_keyword_id": rec.created_keyword_id,
        "created_at": rec.created_at.isoformat() if rec.created_at else None,
        "resolved_at": rec.resolved_at.isoformat() if rec.resolved_at else None,
        "evidence": evidence,
    }


def _serialize_copilot_metadata_variant(variant: AICopilotMetadataVariant):
    return {
        "id": variant.id,
        "run_id": variant.run_id,
        "app_id": variant.app_id,
        "title": variant.title,
        "subtitle": variant.subtitle,
        "keyword_field": variant.keyword_field,
        "covered_keywords": variant.covered_keywords_json or [],
        "predicted_impact": variant.predicted_impact,
        "rationale": variant.rationale,
        "status": variant.status,
        "created_at": variant.created_at.isoformat() if variant.created_at else None,
        "resolved_at": variant.resolved_at.isoformat() if variant.resolved_at else None,
    }


def _serialize_copilot_run(run: AICopilotRun | None):
    if not run:
        return None
    return {
        "id": run.id,
        "status": run.status,
        "progress_percent": int(run.progress_percent or 0),
        "progress_stage": run.progress_stage or "queued",
        "progress_detail": run.progress_detail or "",
        "model": run.model,
        "country": run.country,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "error": run.error,
    }


def _serialize_asc_sync_run(run: ASCSyncRun | None):
    if not run:
        return None
    return {
        "id": run.id,
        "status": run.status,
        "days_back": run.days_back,
        "rows_upserted": run.rows_upserted,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "error": run.error,
    }


def _effective_asc_app_id(app: App) -> str:
    explicit = str(app.asc_app_id or "").strip()
    if explicit:
        return explicit
    if app.track_id:
        return str(app.track_id).strip()
    return ""


def _asc_status_for_app(app: App):
    latest_run = ASCSyncRun.objects.filter(app=app).order_by("-started_at").first()
    latest_metric_date = (
        ASCMetricDaily.objects.filter(app=app).order_by("-date").values_list("date", flat=True).first()
    )
    resolved_asc_app_id = _effective_asc_app_id(app)
    return {
        "app_id": app.id,
        "asc_app_id": resolved_asc_app_id,
        "asc_app_id_source": "asc_app_id" if str(app.asc_app_id or "").strip() else ("track_id" if app.track_id else "none"),
        "latest_run": _serialize_asc_sync_run(latest_run),
        "metric_rows": ASCMetricDaily.objects.filter(app=app).count(),
        "latest_metric_date": latest_metric_date.isoformat() if latest_metric_date else None,
    }


def _copilot_payload_for_app(app: App):
    runs = list(AICopilotRun.objects.filter(app=app).order_by("-started_at")[:20])
    recommendations_qs = (
        AICopilotRecommendation.objects.filter(app=app)
        .select_related("created_keyword", "run")
        .order_by("-created_at")[:300]
    )
    variants_qs = (
        AICopilotMetadataVariant.objects.filter(app=app)
        .select_related("run")
        .order_by("-created_at")[:50]
    )
    return {
        "app": {
            "id": app.id,
            "name": app.name,
            "track_id": app.track_id,
            "asc_app_id": _effective_asc_app_id(app),
            "asc_app_id_source": "asc_app_id" if str(app.asc_app_id or "").strip() else ("track_id" if app.track_id else "none"),
        },
        "runs": [_serialize_copilot_run(item) for item in runs],
        "recommendations": [_serialize_copilot_recommendation(item) for item in recommendations_qs],
        "metadata_variants": [_serialize_copilot_metadata_variant(item) for item in variants_qs],
        "asc_status": _asc_status_for_app(app),
    }


def _get_app_by_id_param(app_id_raw):
    try:
        app_id = int(app_id_raw)
    except (TypeError, ValueError):
        return None
    return App.objects.filter(id=app_id).first()


def _resolve_country_choice(country_raw):
    value = str(country_raw or "").strip().lower()
    valid_codes = {code for code, _ in COUNTRY_CHOICES}
    if not value:
        return "us"
    if value not in valid_codes:
        raise AISuggestionError("Invalid country code.")
    return value


def _parse_csv_models(raw: str) -> list[str]:
    items = []
    for item in str(raw or "").split(","):
        value = item.strip()
        if value and value not in items:
            items.append(value)
    return items


def _effective_ai_settings() -> dict:
    runtime = RuntimeConfig.get_solo()
    api_key = runtime.openai_api_key.strip() or settings.OPENAI_API_KEY
    default_model = runtime.openai_default_model.strip() or settings.OPENAI_MODEL
    model_csv = runtime.openai_available_models.strip()
    if not model_csv:
        model_csv = ",".join(settings.OPENAI_AVAILABLE_MODELS)
    available_models = _parse_csv_models(model_csv)
    if default_model and default_model not in available_models:
        available_models.insert(0, default_model)
    if not available_models and default_model:
        available_models = [default_model]
    system_prompt = runtime.ai_system_prompt.strip() or settings.AI_SYSTEM_PROMPT
    user_prompt_template = runtime.ai_user_prompt_template.strip() or settings.AI_USER_PROMPT_TEMPLATE
    if runtime.ai_enable_online_context is None:
        enable_online_context = settings.AI_ENABLE_ONLINE_CONTEXT
    else:
        enable_online_context = bool(runtime.ai_enable_online_context)
    online_top_apps_per_country = (
        runtime.ai_online_top_apps_per_country
        if runtime.ai_online_top_apps_per_country
        else settings.AI_ONLINE_TOP_APPS_PER_COUNTRY
    )
    online_top_apps_per_country = max(5, min(50, int(online_top_apps_per_country)))
    history_rows_max = (
        runtime.ai_history_rows_max
        if runtime.ai_history_rows_max
        else settings.AI_HISTORY_ROWS_MAX
    )
    history_rows_max = max(50, min(5000, int(history_rows_max)))

    api_key_source = "none"
    if runtime.openai_api_key.strip():
        api_key_source = "config"
    elif settings.OPENAI_API_KEY:
        api_key_source = "env"

    asc_issuer_id = runtime.asc_issuer_id.strip()
    asc_key_id = runtime.asc_key_id.strip()
    asc_private_key = runtime.asc_private_key_pem.strip()
    asc_default_days_back = runtime.asc_default_days_back or settings.ASC_DEFAULT_DAYS_BACK
    asc_default_days_back = max(1, min(365, int(asc_default_days_back)))
    asc_configured = bool(asc_issuer_id and asc_key_id and asc_private_key)
    asc_key_source = "config" if asc_configured else "none"

    return {
        "api_key": api_key,
        "default_model": default_model,
        "available_models": available_models,
        "api_key_source": api_key_source,
        "system_prompt": system_prompt,
        "user_prompt_template": user_prompt_template,
        "enable_online_context": enable_online_context,
        "online_top_apps_per_country": online_top_apps_per_country,
        "history_rows_max": history_rows_max,
        "asc_issuer_id": asc_issuer_id,
        "asc_key_id": asc_key_id,
        "asc_private_key": asc_private_key,
        "asc_default_days_back": asc_default_days_back,
        "asc_configured": asc_configured,
        "asc_key_source": asc_key_source,
    }


def _resolve_model_choice(model_raw, *, default_model: str, allowed_models: list[str]):
    model = str(model_raw or "").strip()
    if not model:
        model = default_model
    if not model:
        raise AISuggestionError("No AI model configured.")
    if allowed_models and model not in allowed_models:
        raise AISuggestionError(
            f"Model '{model}' is not allowed. Configure allowed models on the Config page."
        )
    return model


def ai_suggestions_view(request):
    app_id = request.GET.get("app_id")
    if app_id:
        app = _get_app_by_id_param(app_id)
        if not app:
            return JsonResponse({"error": "Valid app_id is required."}, status=400)
        return JsonResponse(_ai_payload_for_app(app))

    apps = App.objects.all()
    effective = _effective_ai_settings()
    selected_app = request.GET.get("app")
    selected_country_raw = request.GET.get("country")
    try:
        selected_app_id = int(selected_app) if selected_app else None
    except (TypeError, ValueError):
        selected_app_id = None
    try:
        selected_country = _resolve_country_choice(selected_country_raw)
    except AISuggestionError:
        selected_country = "us"
    return render(
        request,
        "aso/ai_suggestions.html",
        {
            "apps": apps,
            "selected_app": selected_app_id,
            "available_models": effective["available_models"],
            "default_model": effective["default_model"],
            "country_choices": COUNTRY_CHOICES,
            "selected_country": selected_country,
            "asc_configured": effective["asc_configured"],
            "asc_default_days_back": effective["asc_default_days_back"],
        },
    )


def ai_suggestions_list_view(request):
    app_id = request.GET.get("app_id")
    if not app_id:
        return JsonResponse({"error": "app_id is required."}, status=400)
    app = _get_app_by_id_param(app_id)
    if not app:
        return JsonResponse({"error": "Valid app_id is required."}, status=400)
    return JsonResponse(_ai_payload_for_app(app))


@require_POST
def ai_suggestions_generate_view(request):
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)
    app_id = body.get("app_id")
    if not app_id:
        return JsonResponse({"error": "app_id is required."}, status=400)
    app = get_object_or_404(App, id=app_id)
    model = body.get("model")
    country = body.get("country")
    effective = _effective_ai_settings()

    try:
        selected_country = _resolve_country_choice(country)
        selected_model = _resolve_model_choice(
            model,
            default_model=effective["default_model"],
            allowed_models=effective["available_models"],
        )
        run = generate_ai_suggestions(
            app,
            model=selected_model,
            api_key=effective["api_key"],
            countries=[selected_country],
            system_prompt=effective["system_prompt"],
            user_prompt_template=effective["user_prompt_template"],
            enable_online_context=effective["enable_online_context"],
            online_top_apps_per_country=effective["online_top_apps_per_country"],
            history_rows_max=effective["history_rows_max"],
        )
    except AISuggestionError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:  # pragma: no cover - defensive runtime path
        logger.exception("AI suggestion generation failed.")
        return JsonResponse({"error": str(exc)}, status=500)

    suggestions = [_serialize_ai_suggestion(s) for s in run.suggestions.order_by("-score_overall")]
    return JsonResponse(
        {
            "success": True,
            "run": {
                "id": run.id,
                "status": run.status,
                "model": run.model,
                "countries": run.countries_json,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                "error": run.error,
            },
            "suggestions": suggestions,
        }
    )


@require_POST
def ai_suggestion_accept_view(request, suggestion_id):
    suggestion = get_object_or_404(AISuggestion, id=suggestion_id)
    suggestion = accept_ai_suggestion(suggestion)
    return JsonResponse({"success": True, "suggestion": _serialize_ai_suggestion(suggestion)})


@require_POST
def ai_suggestion_reject_view(request, suggestion_id):
    suggestion = get_object_or_404(AISuggestion, id=suggestion_id)
    suggestion = reject_ai_suggestion(suggestion)
    return JsonResponse({"success": True, "suggestion": _serialize_ai_suggestion(suggestion)})


def _build_asc_service(effective: dict):
    return AppStoreConnectService(
        issuer_id=effective["asc_issuer_id"],
        key_id=effective["asc_key_id"],
        private_key_pem=effective["asc_private_key"],
    )


@require_POST
def app_store_connect_sync_view(request):
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    app_id = body.get("app_id")
    if not app_id:
        return JsonResponse({"error": "app_id is required."}, status=400)

    app = get_object_or_404(App, id=app_id)
    resolved_asc_app_id = _effective_asc_app_id(app)
    if not resolved_asc_app_id:
        return JsonResponse(
            {"error": "This app is missing asc_app_id and has no track_id fallback."},
            status=400,
        )

    effective = _effective_ai_settings()
    if not effective["asc_configured"]:
        return JsonResponse({"error": "App Store Connect credentials are not configured."}, status=400)

    days_back_raw = body.get("days_back", effective["asc_default_days_back"])
    try:
        days_back = max(1, min(365, int(days_back_raw)))
    except (TypeError, ValueError):
        return JsonResponse({"error": "days_back must be an integer between 1 and 365."}, status=400)

    run = ASCSyncRun.objects.create(
        app=app,
        status=ASCSyncRun.STATUS_RUNNING,
        days_back=days_back,
        rows_upserted=0,
    )
    try:
        service = _build_asc_service(effective)
        rows_upserted = service.sync_app_metrics(app, days_back=days_back, asc_app_id=resolved_asc_app_id)
        run.status = ASCSyncRun.STATUS_SUCCESS
        run.rows_upserted = rows_upserted
        run.finished_at = timezone.now()
        run.error = ""
        run.save(update_fields=["status", "rows_upserted", "finished_at", "error"])
    except (ASCAuthError, ASCError, ASCAPIError) as exc:
        logger.warning(
            "ASC sync failed for app=%s asc_app_id=%s days_back=%s: %s",
            app.id,
            resolved_asc_app_id,
            days_back,
            exc,
        )
        run.status = ASCSyncRun.STATUS_FAILED
        run.rows_upserted = 0
        run.finished_at = timezone.now()
        run.error = str(exc)
        run.save(update_fields=["status", "rows_upserted", "finished_at", "error"])
        return JsonResponse({"error": str(exc), "run": _serialize_asc_sync_run(run)}, status=400)
    except Exception as exc:  # pragma: no cover - defensive runtime path
        logger.exception(
            "ASC sync crashed for app=%s asc_app_id=%s days_back=%s",
            app.id,
            resolved_asc_app_id,
            days_back,
        )
        run.status = ASCSyncRun.STATUS_FAILED
        run.rows_upserted = 0
        run.finished_at = timezone.now()
        run.error = str(exc)
        run.save(update_fields=["status", "rows_upserted", "finished_at", "error"])
        return JsonResponse({"error": str(exc), "run": _serialize_asc_sync_run(run)}, status=500)

    return JsonResponse(
        {
            "success": True,
            "run": _serialize_asc_sync_run(run),
            "rows_upserted": run.rows_upserted,
            "status": _asc_status_for_app(app),
        }
    )


def app_store_connect_status_view(request):
    app_id = request.GET.get("app_id")
    if not app_id:
        return JsonResponse({"error": "app_id is required."}, status=400)
    app = _get_app_by_id_param(app_id)
    if not app:
        return JsonResponse({"error": "Valid app_id is required."}, status=400)
    effective = _effective_ai_settings()
    payload = _asc_status_for_app(app)
    payload["credentials_configured"] = effective["asc_configured"]
    payload["key_source"] = effective["asc_key_source"]
    return JsonResponse(payload)


def ai_copilot_list_view(request):
    app_id = request.GET.get("app_id")
    if not app_id:
        return JsonResponse({"error": "app_id is required."}, status=400)
    app = _get_app_by_id_param(app_id)
    if not app:
        return JsonResponse({"error": "Valid app_id is required."}, status=400)
    return JsonResponse(_copilot_payload_for_app(app))


def ai_copilot_status_view(request):
    app_id = request.GET.get("app_id")
    if not app_id:
        return JsonResponse({"error": "app_id is required."}, status=400)
    app = _get_app_by_id_param(app_id)
    if not app:
        return JsonResponse({"error": "Valid app_id is required."}, status=400)

    run_id_raw = request.GET.get("run_id")
    run = None
    if run_id_raw:
        try:
            run_id = int(run_id_raw)
        except (TypeError, ValueError):
            return JsonResponse({"error": "run_id must be an integer."}, status=400)
        run = AICopilotRun.objects.filter(app=app, id=run_id).first()
    if run is None:
        run = AICopilotRun.objects.filter(app=app).order_by("-started_at").first()

    return JsonResponse(
        {
            "app_id": app.id,
            "run": _serialize_copilot_run(run),
            "has_running_run": bool(run and run.status == AICopilotRun.STATUS_RUNNING),
        }
    )


@require_POST
def ai_copilot_generate_view(request):
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    app_id = body.get("app_id")
    if not app_id:
        return JsonResponse({"error": "app_id is required."}, status=400)
    app = get_object_or_404(App, id=app_id)

    country = body.get("country")
    model = body.get("model")
    sync_asc = bool(body.get("sync_asc", False))
    effective = _effective_ai_settings()

    try:
        selected_country = _resolve_country_choice(country)
        selected_model = _resolve_model_choice(
            model,
            default_model=effective["default_model"],
            allowed_models=effective["available_models"],
        )
        if sync_asc and not effective["asc_configured"]:
            raise AICopilotError("App Store Connect credentials are not configured.")
        resolved_asc_app_id = _effective_asc_app_id(app)
        if sync_asc and not resolved_asc_app_id:
            raise AICopilotError("This app is missing asc_app_id and has no track_id fallback.")
        stale_cutoff = timezone.now() - timedelta(minutes=COPILOT_STALE_MINUTES)
        running_run = AICopilotRun.objects.filter(app=app, status=AICopilotRun.STATUS_RUNNING).order_by("-started_at").first()
        if running_run and running_run.started_at and running_run.started_at < stale_cutoff:
            logger.warning(
                "Marking stale copilot run %s as failed for app=%s (started_at=%s, cutoff=%s)",
                running_run.id,
                app.id,
                running_run.started_at.isoformat(),
                stale_cutoff.isoformat(),
            )
            _mark_copilot_run_failed(
                run_id=running_run.id,
                message=(
                    "Marked failed because an older background run exceeded the stale threshold "
                    f"({COPILOT_STALE_MINUTES}m)."
                ),
            )
            running_run = None

        if running_run:
            return JsonResponse(
                {
                    "error": "A Copilot run is already in progress for this app.",
                    "run": _serialize_copilot_run(running_run),
                    "has_running_run": True,
                },
                status=409,
            )

        run = AICopilotRun.objects.create(
            app=app,
            status=AICopilotRun.STATUS_RUNNING,
            progress_percent=1,
            progress_stage="queued",
            progress_detail="Queued. Starting background copilot run.",
            model=selected_model,
            country=selected_country,
            input_snapshot_json={},
            feature_summary_json={},
        )

        _start_copilot_background_run(
            run_id=run.id,
            app_id=app.id,
            country=selected_country,
            model=selected_model,
            sync_asc=sync_asc,
            effective_settings={
                "api_key": effective["api_key"],
                "system_prompt": effective["system_prompt"],
                "user_prompt_template": effective["user_prompt_template"],
                "asc_key_id": effective["asc_key_id"],
                "asc_issuer_id": effective["asc_issuer_id"],
                "asc_private_key": effective["asc_private_key"],
                "asc_default_days_back": effective["asc_default_days_back"],
            },
        )
    except AISuggestionError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except AICopilotError as exc:
        status_code = 400 if exc.user_error else 503
        return JsonResponse({"error": str(exc)}, status=status_code)
    except Exception as exc:  # pragma: no cover - defensive runtime path
        logger.exception("AI copilot generation failed.")
        return JsonResponse({"error": str(exc)}, status=500)

    return JsonResponse(
        {
            "success": True,
            "started": True,
            "run": _serialize_copilot_run(run),
            "has_running_run": True,
        },
        status=202,
    )


@require_POST
def ai_copilot_recommendation_accept_view(request, recommendation_id):
    recommendation = get_object_or_404(AICopilotRecommendation, id=recommendation_id)
    recommendation = accept_copilot_recommendation(recommendation)
    return JsonResponse({"success": True, "recommendation": _serialize_copilot_recommendation(recommendation)})


@require_POST
def ai_copilot_recommendation_reject_view(request, recommendation_id):
    recommendation = get_object_or_404(AICopilotRecommendation, id=recommendation_id)
    recommendation = reject_copilot_recommendation(recommendation)
    return JsonResponse({"success": True, "recommendation": _serialize_copilot_recommendation(recommendation)})


@require_POST
def ai_copilot_metadata_accept_view(request, variant_id):
    variant = get_object_or_404(AICopilotMetadataVariant, id=variant_id)
    variant = accept_copilot_metadata_variant(variant)
    return JsonResponse({"success": True, "metadata_variant": _serialize_copilot_metadata_variant(variant)})


@require_POST
def ai_copilot_metadata_reject_view(request, variant_id):
    variant = get_object_or_404(AICopilotMetadataVariant, id=variant_id)
    variant = reject_copilot_metadata_variant(variant)
    return JsonResponse({"success": True, "metadata_variant": _serialize_copilot_metadata_variant(variant)})


def version_check_view(request):
    """Check GitHub for a newer release. Returns JSON with update info."""
    current = settings.VERSION
    try:
        url = "https://api.github.com/repos/respectlytics/respectaso/releases/latest"
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        latest = data.get("tag_name", "").lstrip("v")
        if not latest:
            return JsonResponse({"update_available": False, "current": current})
        # Simple semver comparison
        current_parts = [int(x) for x in current.split(".")]
        latest_parts = [int(x) for x in latest.split(".")]
        update_available = latest_parts > current_parts
        return JsonResponse({
            "update_available": update_available,
            "current": current,
            "latest": latest,
            "release_url": data.get("html_url", ""),
        })
    except Exception:
        # Network error, GitHub down, etc. — silently fail
        return JsonResponse({"update_available": False, "current": current})


def auto_refresh_status_view(request):
    """Return the current auto-refresh progress as JSON."""
    return JsonResponse(RefreshRun.latest_status_payload())


def keyword_trend_view(request, keyword_id):
    """
    Return historical trend data for a keyword across all countries.

    Query param: ?country=us (optional, defaults to all)
    Returns JSON with date-series data for charting.
    """
    keyword_obj = get_object_or_404(Keyword, id=keyword_id)
    country = request.GET.get("country")

    qs = SearchResult.objects.filter(keyword=keyword_obj).order_by("searched_at")
    if country:
        qs = qs.filter(country=country)

    data_points = []
    for r in qs:
        data_points.append({
            "date": r.searched_at.strftime("%Y-%m-%d"),
            "datetime": r.searched_at.isoformat(),
            "date_display": r.searched_at.strftime("%b %d, %H:%M"),
            "popularity": r.popularity_score,
            "difficulty": r.difficulty_score,
            "rank": r.app_rank,
            "country": r.country,
        })

    return JsonResponse({
        "keyword": keyword_obj.keyword,
        "keyword_id": keyword_obj.pk,
        "app_name": keyword_obj.app.name if keyword_obj.app else None,
        "data_points": data_points,
    })
