"""
Deterministic feature engineering and scoring for AI Copilot.
"""

from __future__ import annotations

from datetime import timedelta
import math
import re
from typing import Any

from django.db.models import Avg, Max, Sum
from django.utils import timezone

from .models import ASCMetricDaily, App, SearchResult

WORD_RE = re.compile(r"[a-z0-9]+")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_composite_score(
    *,
    score_market: float,
    score_rank_momentum: float,
    score_business_impact: float,
    score_coverage_gap: float,
    llm_confidence: float,
) -> float:
    return _clamp01(
        (0.35 * _clamp01(score_market))
        + (0.20 * _clamp01(score_rank_momentum))
        + (0.20 * _clamp01(score_business_impact))
        + (0.15 * _clamp01(score_coverage_gap))
        + (0.10 * _clamp01(llm_confidence))
    )


def classify_action(score_overall: float) -> str:
    score = _clamp01(score_overall)
    if score >= 0.85:
        return "promote"
    if score >= 0.70:
        return "add"
    if score >= 0.45:
        return "watch"
    return "deprioritize"


def _tokenize(text: str) -> set[str]:
    return set(WORD_RE.findall((text or "").lower()))


def _coverage_gap(keyword: str, historical_keywords: list[str], app_name: str) -> float:
    kw_tokens = _tokenize(keyword)
    if not kw_tokens:
        return 0.5

    best = 0.0
    for hist_kw in historical_keywords[:100]:
        hist_tokens = _tokenize(hist_kw)
        if not hist_tokens:
            continue
        overlap = len(kw_tokens & hist_tokens)
        if not overlap:
            continue
        jaccard = overlap / max(1, len(kw_tokens | hist_tokens))
        best = max(best, jaccard)

    app_tokens = _tokenize(app_name)
    app_overlap = len(kw_tokens & app_tokens) / max(1, len(kw_tokens))
    gap = 1.0 - (0.7 * best + 0.3 * app_overlap)
    return _clamp01(gap)


def _business_impact_from_asc(app: App, country: str) -> tuple[float, dict[str, Any]]:
    cutoff = timezone.now().date() - timedelta(days=30)
    qs = ASCMetricDaily.objects.filter(app=app, country=country, date__gte=cutoff)
    agg = qs.aggregate(
        impressions=Sum("impressions"),
        views=Sum("product_page_views"),
        units=Sum("app_units"),
        conversion=Avg("conversion_rate"),
        proceeds=Sum("proceeds"),
        rows=Max("date"),
    )
    if not qs.exists():
        return 0.25, {
            "has_asc_data": False,
            "impressions_30d": 0,
            "product_page_views_30d": 0,
            "app_units_30d": 0,
            "conversion_rate_avg_30d": None,
            "proceeds_30d": 0.0,
        }

    impressions = int(agg.get("impressions") or 0)
    views = int(agg.get("views") or 0)
    units = int(agg.get("units") or 0)
    conversion = agg.get("conversion")
    proceeds = float(agg.get("proceeds") or 0.0)

    # Log-scaled normalization for heavy-tail app metrics.
    impressions_score = _clamp01(math.log10(impressions + 1) / 6.0)
    views_score = _clamp01(math.log10(views + 1) / 6.0)
    units_score = _clamp01(math.log10(units + 1) / 5.0)
    conversion_score = _clamp01((float(conversion or 0.0)) / 100.0)
    proceeds_score = _clamp01(math.log10(max(proceeds, 0.0) + 1.0) / 5.0)

    score = _clamp01(
        0.25 * impressions_score
        + 0.20 * views_score
        + 0.30 * units_score
        + 0.15 * conversion_score
        + 0.10 * proceeds_score
    )
    return score, {
        "has_asc_data": True,
        "impressions_30d": impressions,
        "product_page_views_30d": views,
        "app_units_30d": units,
        "conversion_rate_avg_30d": conversion,
        "proceeds_30d": proceeds,
    }


def build_keyword_feature_rows(*, app: App, country: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    country = (country or "us").lower()
    rows = list(
        SearchResult.objects.filter(keyword__app=app, country=country)
        .select_related("keyword")
        .order_by("keyword_id", "-searched_at")
    )
    if not rows:
        return [], {"country": country, "count": 0, "asc": {"has_asc_data": False}}

    latest_by_keyword: dict[int, SearchResult] = {}
    prev_by_keyword: dict[int, SearchResult] = {}
    for row in rows:
        if row.keyword_id not in latest_by_keyword:
            latest_by_keyword[row.keyword_id] = row
        elif row.keyword_id not in prev_by_keyword:
            prev_by_keyword[row.keyword_id] = row

    opportunities: dict[int, float] = {}
    momentum_values: dict[int, float] = {}
    historical_keywords = [r.keyword.keyword for r in latest_by_keyword.values()]
    for keyword_id, row in latest_by_keyword.items():
        pop = row.popularity_score or 0
        diff = row.difficulty_score or 0
        opportunities[keyword_id] = max(0.0, min(100.0, pop * (100 - diff) / 100))
        prev = prev_by_keyword.get(keyword_id)
        if prev and row.app_rank and prev.app_rank:
            raw = prev.app_rank - row.app_rank  # positive means improvement
            momentum_values[keyword_id] = _clamp01((raw + 10.0) / 20.0)
        elif row.app_rank:
            momentum_values[keyword_id] = 0.55
        else:
            momentum_values[keyword_id] = 0.4

    max_opportunity = max(opportunities.values()) if opportunities else 1.0
    business_impact_score, asc_summary = _business_impact_from_asc(app, country)

    feature_rows = []
    for keyword_id, row in latest_by_keyword.items():
        score_market = _clamp01(opportunities[keyword_id] / max(1.0, max_opportunity))
        score_rank = _clamp01(momentum_values[keyword_id])
        score_coverage = _coverage_gap(row.keyword.keyword, historical_keywords, app.name)
        feature_rows.append(
            {
                "keyword_id": row.keyword_id,
                "keyword": row.keyword.keyword,
                "country": country,
                "popularity": row.popularity_score,
                "difficulty": row.difficulty_score,
                "opportunity": opportunities[keyword_id],
                "app_rank": row.app_rank,
                "prev_app_rank": prev_by_keyword.get(keyword_id).app_rank if prev_by_keyword.get(keyword_id) else None,
                "score_market_opportunity": score_market,
                "score_rank_momentum": score_rank,
                "score_business_impact": business_impact_score,
                "score_coverage_gap": score_coverage,
                "evidence": {
                    "latest_result_id": row.id,
                    "latest_searched_at": row.searched_at.isoformat() if row.searched_at else None,
                    "asc_summary": asc_summary,
                },
            }
        )

    feature_rows.sort(
        key=lambda item: (
            item["score_market_opportunity"],
            item["score_rank_momentum"],
            item["score_coverage_gap"],
        ),
        reverse=True,
    )

    summary = {
        "country": country,
        "count": len(feature_rows),
        "asc": asc_summary,
        "top_keywords": [row["keyword"] for row in feature_rows[:20]],
    }
    return feature_rows, summary


def merge_candidate_confidence(
    rows: list[dict[str, Any]],
    confidence_by_keyword: dict[str, float],
) -> list[dict[str, Any]]:
    merged = []
    for row in rows:
        llm_confidence = _clamp01(confidence_by_keyword.get(row["keyword"], 0.5))
        score_overall = compute_composite_score(
            score_market=row["score_market_opportunity"],
            score_rank_momentum=row["score_rank_momentum"],
            score_business_impact=row["score_business_impact"],
            score_coverage_gap=row["score_coverage_gap"],
            llm_confidence=llm_confidence,
        )
        merged_row = row.copy()
        merged_row["llm_confidence"] = llm_confidence
        merged_row["score_overall"] = score_overall
        merged_row["action"] = classify_action(score_overall)
        merged.append(merged_row)

    merged.sort(key=lambda item: item["score_overall"], reverse=True)
    return merged
