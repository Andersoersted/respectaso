"""
OpenAI-backed keyword suggestion pipeline (Apple App Store only).
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import timedelta

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from .models import AISuggestion, AISuggestionRun, App, Keyword, SearchResult
from .services import (
    DifficultyCalculator,
    DownloadEstimator,
    ITunesSearchService,
    PopularityEstimator,
)

logger = logging.getLogger(__name__)

WORD_RE = re.compile(r"[a-z0-9]+")


class AISuggestionError(Exception):
    """Raised when suggestion generation cannot continue safely."""


def _tokenize(text: str) -> set[str]:
    return set(WORD_RE.findall((text or "").lower()))


def _build_openai_client(api_key: str):
    if not api_key:
        raise AISuggestionError("OPENAI_API_KEY is not configured.")
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - import guard
        raise AISuggestionError("OpenAI SDK is not installed. Add `openai` to requirements.") from exc
    return OpenAI(
        api_key=api_key,
        timeout=settings.OPENAI_TIMEOUT_SECONDS,
        max_retries=settings.OPENAI_MAX_RETRIES,
    )


def _choose_countries(app: App) -> list[str]:
    countries = list(
        SearchResult.objects.filter(keyword__app=app)
        .values_list("country", flat=True)
        .distinct()
        .order_by("country")
    )
    if not countries:
        countries = ["us"]
    return countries[: settings.AI_MAX_COUNTRIES]


def _build_history_summary(app: App) -> tuple[list[dict], list[str]]:
    """
    Return top historical keywords by opportunity + a list for lexical overlap.
    """
    cutoff = timezone.now() - timedelta(days=30)
    history = (
        SearchResult.objects.filter(keyword__app=app, searched_at__gte=cutoff)
        .select_related("keyword")
        .order_by("-searched_at")
    )

    latest_by_key_country = {}
    for row in history:
        key = (row.keyword_id, row.country)
        if key not in latest_by_key_country:
            latest_by_key_country[key] = row

    scored = []
    keyword_scores = defaultdict(list)
    for row in latest_by_key_country.values():
        pop = row.popularity_score or 0
        diff = row.difficulty_score or 0
        opportunity = round(pop * (100 - diff) / 100)
        keyword_scores[row.keyword.keyword].append(opportunity)
        scored.append(
            {
                "keyword": row.keyword.keyword,
                "country": row.country,
                "popularity": pop,
                "difficulty": diff,
                "opportunity": opportunity,
            }
        )
    scored.sort(key=lambda item: item["opportunity"], reverse=True)
    top_keywords = sorted(
        ((kw, sum(vals) / max(len(vals), 1)) for kw, vals in keyword_scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    historical_keyword_list = [item[0] for item in top_keywords[:30]]
    return scored[:30], historical_keyword_list


def _build_input_snapshot(app: App, countries: list[str]) -> dict:
    itunes_service = ITunesSearchService()
    app_lookup = itunes_service.lookup_by_id(app.track_id, country=countries[0]) if app.track_id else None

    existing_keywords = list(
        Keyword.objects.filter(app=app).values_list("keyword", flat=True).order_by("-created_at")
    )
    history_rows, historical_keyword_list = _build_history_summary(app)

    snapshot = {
        "app": {
            "name": app.name,
            "bundle_id": app.bundle_id,
            "track_id": app.track_id,
            "seller_name": app.seller_name,
            "store_metadata": app_lookup or {},
        },
        "countries": countries,
        "existing_keywords": existing_keywords[:100],
        "history_top_rows": history_rows,
    }
    return snapshot | {"history_keywords": historical_keyword_list}


def _json_schema():
    return {
        "name": "keyword_candidates",
        "description": "Apple App Store keyword candidates for one app",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["candidates"],
            "properties": {
                "candidates": {
                    "type": "array",
                    "maxItems": 30,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["keyword", "intent", "rationale", "relevance_confidence"],
                        "properties": {
                            "keyword": {"type": "string"},
                            "intent": {"type": "string"},
                            "rationale": {"type": "string"},
                            "relevance_confidence": {"type": "number"},
                        },
                    },
                }
            },
        },
    }


def _generate_candidates(snapshot: dict, model: str, api_key: str) -> list[dict]:
    client = _build_openai_client(api_key)
    prompt = (
        "You are an App Store Optimization analyst for Apple App Store only.\n"
        "Generate candidate keywords for this exact app.\n"
        "Prioritize: intent relevance, realistic competition, and discoverability.\n"
        "Avoid generic junk or duplicate phrases.\n"
        "Input data JSON:\n"
        f"{json.dumps(snapshot, ensure_ascii=True)}"
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return JSON that exactly matches the schema. "
                    "Focus on Apple App Store keyword recommendations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": _json_schema(),
        },
    )
    content = ""
    if response.choices:
        content = response.choices[0].message.content or ""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AISuggestionError("OpenAI returned non-JSON content for suggestions.") from exc

    candidates = parsed.get("candidates")
    if not isinstance(candidates, list):
        raise AISuggestionError("OpenAI response schema was invalid (missing candidates array).")
    return candidates


def _clean_candidate_keyword(raw: str) -> str:
    kw = (raw or "").strip().lower()
    kw = re.sub(r"\s+", " ", kw)
    return kw


def _dedupe_candidates(candidates: list[dict], existing_keywords: set[str]) -> list[dict]:
    seen = set()
    cleaned = []
    for item in candidates:
        keyword = _clean_candidate_keyword(item.get("keyword", ""))
        if not keyword or len(keyword) < 2 or len(keyword) > 80:
            continue
        if keyword in existing_keywords or keyword in seen:
            continue
        confidence = item.get("relevance_confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        seen.add(keyword)
        cleaned.append(
            {
                "keyword": keyword,
                "intent": str(item.get("intent", "")).strip(),
                "rationale": str(item.get("rationale", "")).strip(),
                "relevance_confidence": confidence,
            }
        )
    cleaned.sort(key=lambda item: item["relevance_confidence"], reverse=True)
    return cleaned[: settings.AI_MAX_CANDIDATES]


def _history_score(keyword: str, historical_keywords: list[str]) -> float:
    if not historical_keywords:
        return 0.5
    kw_tokens = _tokenize(keyword)
    if not kw_tokens:
        return 0.5
    best = 0.0
    for hist_kw in historical_keywords[:50]:
        hist_tokens = _tokenize(hist_kw)
        if not hist_tokens:
            continue
        overlap = len(kw_tokens & hist_tokens)
        if overlap == 0:
            continue
        score = overlap / len(kw_tokens | hist_tokens)
        if score > best:
            best = score
    return max(0.0, min(1.0, best if best > 0 else 0.1))


def _evaluate_keyword(
    *,
    app: App,
    keyword: str,
    countries: list[str],
    itunes_service: ITunesSearchService,
    difficulty_calc: DifficultyCalculator,
    popularity_est: PopularityEstimator,
    download_est: DownloadEstimator,
) -> dict:
    per_country = []
    for country in countries:
        competitors = itunes_service.search_apps(keyword, country=country, limit=25)
        difficulty_score, breakdown = difficulty_calc.calculate(competitors, keyword=keyword)
        popularity = popularity_est.estimate(competitors, keyword)
        breakdown["download_estimates"] = download_est.estimate(popularity or 0, len(competitors))
        app_rank = None
        if app.track_id:
            app_rank = itunes_service.find_app_rank(keyword, app.track_id, country=country)
        opportunity = round((popularity or 0) * (100 - difficulty_score) / 100)
        per_country.append(
            {
                "country": country,
                "popularity": popularity,
                "difficulty": difficulty_score,
                "opportunity": opportunity,
                "app_rank": app_rank,
                "difficulty_breakdown": breakdown,
                "competitors_data": competitors,
            }
        )
    market_score = 0.0
    if per_country:
        market_score = sum(item["opportunity"] for item in per_country) / (len(per_country) * 100)
    return {
        "countries": per_country,
        "score_market": max(0.0, min(1.0, market_score)),
    }


@transaction.atomic
def generate_ai_suggestions(
    app: App,
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> AISuggestionRun:
    if not app.track_id:
        raise AISuggestionError("This app must be linked via App Store lookup (track_id) before AI suggestions.")
    chosen_model = (model or settings.OPENAI_MODEL).strip()
    if not chosen_model:
        raise AISuggestionError("No AI model selected.")
    chosen_api_key = (api_key or settings.OPENAI_API_KEY).strip()

    countries = _choose_countries(app)
    snapshot = _build_input_snapshot(app, countries)

    run = AISuggestionRun.objects.create(
        app=app,
        status=AISuggestionRun.STATUS_RUNNING,
        model=chosen_model,
        countries_json=countries,
        input_snapshot_json=snapshot,
    )

    try:
        raw_candidates = _generate_candidates(snapshot, chosen_model, chosen_api_key)
        existing_keywords = {
            _clean_candidate_keyword(value)
            for value in Keyword.objects.filter(app=app).values_list("keyword", flat=True)
        }
        candidates = _dedupe_candidates(raw_candidates, existing_keywords)
        candidates = candidates[: settings.AI_EVALUATED_CANDIDATES]
        if not candidates:
            raise AISuggestionError("No valid keyword candidates were generated.")

        itunes_service = ITunesSearchService()
        difficulty_calc = DifficultyCalculator()
        popularity_est = PopularityEstimator()
        download_est = DownloadEstimator()

        _, historical_keywords = _build_history_summary(app)
        suggestions_to_create = []
        for candidate in candidates:
            eval_result = _evaluate_keyword(
                app=app,
                keyword=candidate["keyword"],
                countries=countries,
                itunes_service=itunes_service,
                difficulty_calc=difficulty_calc,
                popularity_est=popularity_est,
                download_est=download_est,
            )
            score_market = eval_result["score_market"]
            score_history = _history_score(candidate["keyword"], historical_keywords)
            score_overall = (
                0.55 * score_market
                + 0.25 * candidate["relevance_confidence"]
                + 0.20 * score_history
            )
            suggestions_to_create.append(
                AISuggestion(
                    run=run,
                    app=app,
                    keyword=candidate["keyword"],
                    intent=candidate["intent"],
                    rationale=candidate["rationale"],
                    confidence=candidate["relevance_confidence"],
                    score_market=score_market,
                    score_history=score_history,
                    score_overall=score_overall,
                    market_metrics_json=eval_result,
                    status=AISuggestion.STATUS_DRAFT,
                )
            )

        AISuggestion.objects.bulk_create(suggestions_to_create)
        run.status = AISuggestionRun.STATUS_SUCCESS
        run.finished_at = timezone.now()
        run.error = ""
        run.save(update_fields=["status", "finished_at", "error"])
        return run
    except Exception as exc:
        run.status = AISuggestionRun.STATUS_FAILED
        run.finished_at = timezone.now()
        run.error = str(exc)
        run.save(update_fields=["status", "finished_at", "error"])
        raise


@transaction.atomic
def accept_ai_suggestion(suggestion: AISuggestion) -> AISuggestion:
    if suggestion.status == AISuggestion.STATUS_ACCEPTED:
        return suggestion
    app = suggestion.app
    keyword_obj, _ = Keyword.objects.get_or_create(keyword=suggestion.keyword, app=app)
    metrics = suggestion.market_metrics_json or {}
    for item in metrics.get("countries", []):
        SearchResult.create_snapshot(
            keyword=keyword_obj,
            country=item.get("country", "us"),
            source=SearchResult.SOURCE_AI_SUGGESTION,
            popularity_score=item.get("popularity"),
            difficulty_score=item.get("difficulty", 0),
            difficulty_breakdown=item.get("difficulty_breakdown", {}),
            competitors_data=item.get("competitors_data", []),
            app_rank=item.get("app_rank"),
        )
    suggestion.status = AISuggestion.STATUS_ACCEPTED
    suggestion.created_keyword = keyword_obj
    suggestion.resolved_at = timezone.now()
    suggestion.save(update_fields=["status", "created_keyword", "resolved_at"])
    return suggestion


@transaction.atomic
def reject_ai_suggestion(suggestion: AISuggestion) -> AISuggestion:
    if suggestion.status == AISuggestion.STATUS_REJECTED:
        return suggestion
    suggestion.status = AISuggestion.STATUS_REJECTED
    suggestion.resolved_at = timezone.now()
    suggestion.save(update_fields=["status", "resolved_at"])
    return suggestion
