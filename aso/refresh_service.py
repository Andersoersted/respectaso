"""
Refresh orchestration for tracked keyword-country pairs.

Used by both the external cron management command and the optional in-process
thread scheduler fallback.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import timedelta

from django.conf import settings
from django.utils import timezone

from .models import Keyword, RefreshRun, SearchResult
from .services import (
    DifficultyCalculator,
    DownloadEstimator,
    ITunesSearchService,
    PopularityEstimator,
)

logger = logging.getLogger(__name__)


def _today_window():
    start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start, end


def distinct_keyword_country_pairs() -> list[tuple[int, str]]:
    pairs = (
        SearchResult.objects.values_list("keyword_id", "country")
        .distinct()
        .order_by("keyword_id", "country")
    )
    return list(pairs)


def get_pairs_to_refresh(*, force: bool = False) -> list[tuple[int, str]]:
    """
    Return tracked (keyword_id, country) pairs that should be refreshed now.

    A pair is considered done for the day if a `daily_refresh` snapshot exists.
    """
    pairs = distinct_keyword_country_pairs()
    if force or not pairs:
        return pairs

    today_start, today_end = _today_window()
    refreshed_today = set(
        SearchResult.objects.filter(
            source=SearchResult.SOURCE_DAILY_REFRESH,
            searched_at__gte=today_start,
            searched_at__lt=today_end,
        ).values_list("keyword_id", "country")
    )
    return [pair for pair in pairs if pair not in refreshed_today]


def cleanup_old_results() -> int:
    cutoff = timezone.now() - timedelta(days=settings.RESULT_RETENTION_DAYS)
    deleted, _ = SearchResult.objects.filter(searched_at__lt=cutoff).delete()
    if deleted:
        logger.info(
            "Cleaned up %s search result rows older than %s days.",
            deleted,
            settings.RESULT_RETENTION_DAYS,
        )
    return deleted


def refresh_pair(
    *,
    keyword_obj: Keyword,
    country: str,
    source: str,
    itunes_service: ITunesSearchService,
    difficulty_calc: DifficultyCalculator,
    popularity_est: PopularityEstimator,
    download_est: DownloadEstimator,
) -> SearchResult:
    competitors = itunes_service.search_apps(keyword_obj.keyword, country=country, limit=25)
    difficulty_score, breakdown = difficulty_calc.calculate(competitors, keyword=keyword_obj.keyword)

    app_rank = None
    if keyword_obj.app and keyword_obj.app.track_id:
        app_rank = itunes_service.find_app_rank(
            keyword_obj.keyword,
            keyword_obj.app.track_id,
            country=country,
        )

    popularity = popularity_est.estimate(competitors, keyword_obj.keyword)
    breakdown["download_estimates"] = download_est.estimate(popularity or 0, len(competitors))

    return SearchResult.create_snapshot(
        keyword=keyword_obj,
        country=country,
        source=source,
        popularity_score=popularity,
        difficulty_score=difficulty_score,
        difficulty_breakdown=breakdown,
        competitors_data=competitors,
        app_rank=app_rank,
    )


def run_refresh(
    *,
    trigger: str,
    source: str = SearchResult.SOURCE_DAILY_REFRESH,
    force: bool = False,
    throttle_seconds: float = 2.0,
    pairs_override: list[tuple[int, str]] | None = None,
    run_id: int | None = None,
) -> RefreshRun:
    """
    Refresh tracked keyword-country pairs and persist progress in RefreshRun.
    """
    if pairs_override is None:
        pairs = get_pairs_to_refresh(force=force)
    else:
        # De-duplicate while preserving caller ordering.
        seen: set[tuple[int, str]] = set()
        pairs: list[tuple[int, str]] = []
        for keyword_id, country in pairs_override:
            pair = (int(keyword_id), str(country or "us").lower())
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)

    if run_id is None:
        run = RefreshRun.objects.create(
            trigger=trigger,
            status=RefreshRun.STATUS_RUNNING,
            total=len(pairs),
            completed=0,
            current_keyword="",
            error="",
        )
    else:
        run = RefreshRun.objects.get(pk=run_id)
        run.trigger = trigger
        run.status = RefreshRun.STATUS_RUNNING
        run.total = len(pairs)
        run.completed = 0
        run.current_keyword = ""
        run.finished_at = None
        run.error = ""
        run.save(
            update_fields=[
                "trigger",
                "status",
                "total",
                "completed",
                "current_keyword",
                "finished_at",
                "error",
            ]
        )

    if not pairs:
        run.status = RefreshRun.STATUS_SUCCESS
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "finished_at"])
        cleanup_old_results()
        return run

    keyword_map = {
        kw.id: kw
        for kw in Keyword.objects.filter(id__in=[kw_id for kw_id, _ in pairs]).select_related("app")
    }

    itunes_service = ITunesSearchService()
    difficulty_calc = DifficultyCalculator()
    popularity_est = PopularityEstimator()
    download_est = DownloadEstimator()

    failures: dict[str, int] = defaultdict(int)

    for i, (keyword_id, country) in enumerate(pairs):
        keyword_obj = keyword_map.get(keyword_id)
        if keyword_obj is None:
            run.completed = i + 1
            run.save(update_fields=["completed"])
            continue

        run.current_keyword = f"{keyword_obj.keyword} ({country.upper()})"
        run.completed = i
        run.save(update_fields=["current_keyword", "completed"])

        try:
            if i > 0 and throttle_seconds > 0:
                time.sleep(throttle_seconds)
            refresh_pair(
                keyword_obj=keyword_obj,
                country=country,
                source=source,
                itunes_service=itunes_service,
                difficulty_calc=difficulty_calc,
                popularity_est=popularity_est,
                download_est=download_est,
            )
        except Exception as exc:  # pragma: no cover - defensive production path
            key = str(exc)
            failures[key] += 1
            logger.warning(
                "Refresh failed for %s (%s): %s",
                keyword_obj.keyword,
                country,
                exc,
            )

    run.completed = len(pairs)
    run.current_keyword = ""
    run.finished_at = timezone.now()
    if failures:
        run.status = RefreshRun.STATUS_FAILED
        total_failed = sum(failures.values())
        top_message = next(iter(failures.keys()))
        run.error = f"{total_failed} pair(s) failed. First error: {top_message}"
    else:
        run.status = RefreshRun.STATUS_SUCCESS
        run.error = ""
    run.save(
        update_fields=[
            "completed",
            "current_keyword",
            "finished_at",
            "status",
            "error",
        ]
    )

    cleanup_old_results()
    return run
