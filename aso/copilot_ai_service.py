"""
AI Copilot generation + action handlers.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from pydantic import BaseModel, Field

from .copilot_features import (
    build_keyword_feature_rows,
    classify_action,
    compute_composite_score,
)
from .models import (
    AICopilotMetadataVariant,
    AICopilotRecommendation,
    AICopilotRun,
    App,
    Keyword,
    SearchResult,
)

logger = logging.getLogger(__name__)

WORD_RE = re.compile(r"[a-z0-9]+")


class AICopilotError(Exception):
    """Raised when AI Copilot cannot complete safely."""


class RecommendationOut(BaseModel):
    keyword: str = Field(min_length=2, max_length=80)
    action: str = Field(default="watch")
    rationale: str = Field(default="")
    llm_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class MetadataVariantOut(BaseModel):
    title: str = Field(min_length=1, max_length=60)
    subtitle: str = Field(default="", max_length=60)
    keyword_field: str = Field(default="", max_length=180)
    covered_keywords: list[str] = Field(default_factory=list)
    predicted_impact: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = Field(default="")


class CopilotOutput(BaseModel):
    recommendations: list[RecommendationOut] = Field(default_factory=list)
    metadata_variants: list[MetadataVariantOut] = Field(default_factory=list)


@dataclass(slots=True)
class CopilotSettings:
    model: str
    api_key: str
    system_prompt: str
    user_prompt_template: str


def _build_openai_client(api_key: str):
    if not api_key:
        raise AICopilotError("OPENAI_API_KEY is not configured.")
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - import guard
        raise AICopilotError("OpenAI SDK is not installed. Add `openai` to requirements.") from exc

    return OpenAI(
        api_key=api_key,
        timeout=settings.OPENAI_TIMEOUT_SECONDS,
        max_retries=settings.OPENAI_MAX_RETRIES,
    )


def _clean_keyword(value: str) -> str:
    kw = (value or "").strip().lower()
    kw = re.sub(r"\s+", " ", kw)
    return kw


def _keyword_tokens(text: str) -> list[str]:
    return WORD_RE.findall((text or "").lower())


def _dedupe_keyword_field(keyword_field: str) -> str:
    seen = set()
    tokens = []
    for token in [part.strip().lower() for part in (keyword_field or "").split(",") if part.strip()]:
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return ",".join(tokens)


def _truncate_chars(text: str, limit: int) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].strip()


def _render_prompt(template: str, payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=True)
    prompt = (template or "").replace("{{SNAPSHOT_JSON}}", payload_json)
    if "{{SNAPSHOT_JSON}}" not in (template or ""):
        prompt += f"\n\nCOPILOT SNAPSHOT JSON:\n{payload_json}"
    return prompt


def _extract_responses_parse(output: Any) -> CopilotOutput | None:
    parsed = getattr(output, "output_parsed", None)
    if isinstance(parsed, CopilotOutput):
        return parsed
    items = getattr(output, "output", None)
    if not isinstance(items, list):
        return None
    for item in items:
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            continue
        for block in content:
            parsed_block = getattr(block, "parsed", None)
            if isinstance(parsed_block, CopilotOutput):
                return parsed_block
    return None


def _generate_with_openai(
    *,
    snapshot: dict[str, Any],
    settings_obj: CopilotSettings,
) -> CopilotOutput:
    client = _build_openai_client(settings_obj.api_key)
    user_prompt = _render_prompt(settings_obj.user_prompt_template, snapshot)
    model = settings_obj.model

    try:
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": settings_obj.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=CopilotOutput,
        )
        parsed = _extract_responses_parse(response)
        if parsed is not None:
            return parsed
    except Exception as exc:
        logger.warning("responses.parse failed for model %s: %s", model, exc)

    # Fallback for models/routes without responses structured parsing.
    try:
        completion = client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": settings_obj.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=CopilotOutput,
        )
    except Exception as exc:  # pragma: no cover - runtime fallback path
        raise AICopilotError(f"Copilot generation failed: {exc}") from exc

    message = completion.choices[0].message if completion.choices else None
    parsed = getattr(message, "parsed", None) if message else None
    if not isinstance(parsed, CopilotOutput):
        refusal = getattr(message, "refusal", "") if message else ""
        raise AICopilotError(refusal or "Copilot model returned no structured output.")
    return parsed


def _normalize_action(value: str, score_overall: float) -> str:
    raw = str(value or "").strip().lower()
    valid = {
        AICopilotRecommendation.ACTION_ADD,
        AICopilotRecommendation.ACTION_PROMOTE,
        AICopilotRecommendation.ACTION_WATCH,
        AICopilotRecommendation.ACTION_DEPRIORITIZE,
    }
    if raw in valid:
        return raw
    return classify_action(score_overall)


def _build_snapshot(app: App, country: str, feature_rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    existing_keywords = list(
        Keyword.objects.filter(app=app).values_list("keyword", flat=True).order_by("-created_at")[:300]
    )
    return {
        "app": {
            "id": app.id,
            "name": app.name,
            "bundle_id": app.bundle_id,
            "track_id": app.track_id,
            "asc_app_id": app.asc_app_id,
            "seller_name": app.seller_name,
        },
        "country": country,
        "existing_keywords": existing_keywords,
        "feature_summary": summary,
        "feature_rows": feature_rows[:120],
        "rules": {
            "title_max_chars": 30,
            "subtitle_max_chars": 30,
            "keyword_field_max_chars": 100,
            "metadata_variants_target": 3,
        },
    }


def _update_run_fields(run: AICopilotRun, **fields):
    if not fields:
        return
    AICopilotRun.objects.filter(id=run.id).update(**fields)
    for key, value in fields.items():
        setattr(run, key, value)


def _set_run_progress(run: AICopilotRun, *, percent: int, stage: str, detail: str):
    _update_run_fields(
        run,
        progress_percent=max(0, min(100, int(percent))),
        progress_stage=(stage or "running")[:64],
        progress_detail=(detail or "")[:255],
    )


def _mark_run_failed(run: AICopilotRun, error: str):
    _update_run_fields(
        run,
        status=AICopilotRun.STATUS_FAILED,
        finished_at=timezone.now(),
        error=str(error or ""),
        progress_percent=max(5, int(run.progress_percent or 0)),
        progress_stage="failed",
        progress_detail=_truncate_chars(str(error or "Copilot run failed."), 255),
    )


def generate_ai_copilot(
    app: App,
    *,
    country: str,
    settings_obj: CopilotSettings,
) -> AICopilotRun:
    country = (country or "us").lower()
    run = AICopilotRun.objects.create(
        app=app,
        status=AICopilotRun.STATUS_RUNNING,
        progress_percent=2,
        progress_stage="preparing",
        progress_detail="Validating inputs and preparing run.",
        model=settings_obj.model,
        country=country,
        input_snapshot_json={},
        feature_summary_json={},
    )

    try:
        _set_run_progress(
            run,
            percent=12,
            stage="collecting_signals",
            detail="Collecting keyword history and ASC analytics signals.",
        )
        feature_rows, feature_summary = build_keyword_feature_rows(app=app, country=country)
        if not feature_rows:
            raise AICopilotError("No tracked keyword history exists for this app and country.")

        snapshot = _build_snapshot(app, country, feature_rows, feature_summary)
        _update_run_fields(
            run,
            input_snapshot_json=snapshot,
            feature_summary_json=feature_summary,
        )

        _set_run_progress(
            run,
            percent=34,
            stage="generating_recommendations",
            detail="Requesting structured recommendations from the model.",
        )
        output = _generate_with_openai(snapshot=snapshot, settings_obj=settings_obj)

        _set_run_progress(
            run,
            percent=58,
            stage="scoring_recommendations",
            detail="Scoring and deduplicating recommendation candidates.",
        )
        base_rows = {row["keyword"]: row for row in feature_rows}
        business_default = feature_rows[0]["score_business_impact"] if feature_rows else 0.25
        seen_keywords = set()

        rec_objects: list[AICopilotRecommendation] = []
        for rec in output.recommendations[:40]:
            keyword = _clean_keyword(rec.keyword)
            if not keyword or keyword in seen_keywords:
                continue
            seen_keywords.add(keyword)
            row = base_rows.get(
                keyword,
                {
                    "keyword": keyword,
                    "country": country,
                    "score_market_opportunity": 0.5,
                    "score_rank_momentum": 0.45,
                    "score_business_impact": business_default,
                    "score_coverage_gap": 0.65,
                    "evidence": {},
                    "popularity": None,
                    "difficulty": None,
                    "opportunity": None,
                    "app_rank": None,
                    "prev_app_rank": None,
                },
            )

            score_overall = compute_composite_score(
                score_market=row["score_market_opportunity"],
                score_rank_momentum=row["score_rank_momentum"],
                score_business_impact=row["score_business_impact"],
                score_coverage_gap=row["score_coverage_gap"],
                llm_confidence=rec.llm_confidence,
            )

            rec_objects.append(
                AICopilotRecommendation(
                    run=run,
                    app=app,
                    keyword=keyword,
                    action=_normalize_action(rec.action, score_overall),
                    rationale=(rec.rationale or "").strip(),
                    llm_confidence=max(0.0, min(1.0, float(rec.llm_confidence))),
                    score_market=row["score_market_opportunity"],
                    score_rank_momentum=row["score_rank_momentum"],
                    score_business_impact=row["score_business_impact"],
                    score_coverage_gap=row["score_coverage_gap"],
                    score_overall=score_overall,
                    evidence_json={
                        "country": row["country"],
                        "popularity": row.get("popularity"),
                        "difficulty": row.get("difficulty"),
                        "opportunity": row.get("opportunity"),
                        "app_rank": row.get("app_rank"),
                        "prev_app_rank": row.get("prev_app_rank"),
                        "signals": row.get("evidence", {}),
                    },
                    status=AICopilotRecommendation.STATUS_DRAFT,
                )
            )

        if not rec_objects:
            raise AICopilotError("Copilot returned no usable recommendations.")
        rec_objects.sort(key=lambda item: item.score_overall, reverse=True)

        variant_objects: list[AICopilotMetadataVariant] = []
        for variant in output.metadata_variants[:6]:
            title = _truncate_chars(variant.title, 30)
            subtitle = _truncate_chars(variant.subtitle, 30)
            keyword_field = _truncate_chars(_dedupe_keyword_field(variant.keyword_field), 100)
            if not title:
                continue
            if title.lower() == subtitle.lower():
                subtitle = ""
            variant_objects.append(
                AICopilotMetadataVariant(
                    run=run,
                    app=app,
                    title=title,
                    subtitle=subtitle,
                    keyword_field=keyword_field,
                    covered_keywords_json=[_clean_keyword(k) for k in variant.covered_keywords if _clean_keyword(k)],
                    predicted_impact=max(0.0, min(1.0, float(variant.predicted_impact))),
                    rationale=(variant.rationale or "").strip(),
                    status=AICopilotMetadataVariant.STATUS_DRAFT,
                )
            )

        if not variant_objects:
            top_keywords = [obj.keyword for obj in rec_objects[:6]]
            variant_objects = [
                AICopilotMetadataVariant(
                    run=run,
                    app=app,
                    title=_truncate_chars(app.name, 30) or "App",
                    subtitle=_truncate_chars("ASO optimized listing", 30),
                    keyword_field=_truncate_chars(_dedupe_keyword_field(",".join(top_keywords)), 100),
                    covered_keywords_json=top_keywords,
                    predicted_impact=0.5,
                    rationale="Fallback variant generated from top copilot recommendations.",
                    status=AICopilotMetadataVariant.STATUS_DRAFT,
                )
            ]
        variant_objects.sort(key=lambda item: item.predicted_impact, reverse=True)

        _set_run_progress(
            run,
            percent=82,
            stage="persisting_output",
            detail="Saving recommendations and metadata variants.",
        )
        with transaction.atomic():
            AICopilotRecommendation.objects.bulk_create(rec_objects[:30])
            AICopilotMetadataVariant.objects.bulk_create(variant_objects[:3])
            _update_run_fields(
                run,
                status=AICopilotRun.STATUS_SUCCESS,
                finished_at=timezone.now(),
                error="",
                progress_percent=100,
                progress_stage="complete",
                progress_detail="Copilot run completed successfully.",
            )
        return run
    except AICopilotError as exc:
        _mark_run_failed(run, str(exc))
        logger.warning("AI copilot run %s failed for app %s: %s", run.id, app.id, exc)
        raise
    except Exception as exc:
        _mark_run_failed(run, str(exc))
        logger.exception("AI copilot run %s failed unexpectedly for app %s", run.id, app.id)
        raise AICopilotError(str(exc)) from exc


@transaction.atomic
def accept_copilot_recommendation(reco: AICopilotRecommendation) -> AICopilotRecommendation:
    if reco.status == AICopilotRecommendation.STATUS_ACCEPTED:
        return reco

    app = reco.app
    keyword_obj, _ = Keyword.objects.get_or_create(keyword=reco.keyword, app=app)
    country = reco.run.country or "us"
    latest = (
        SearchResult.objects.filter(keyword=keyword_obj, country=country)
        .order_by("-searched_at")
        .first()
    )
    if latest:
        SearchResult.create_snapshot(
            keyword=keyword_obj,
            country=country,
            source=SearchResult.SOURCE_AI_COPILOT,
            popularity_score=latest.popularity_score,
            difficulty_score=latest.difficulty_score,
            difficulty_breakdown=latest.difficulty_breakdown or {},
            competitors_data=latest.competitors_data or [],
            app_rank=latest.app_rank,
        )

    reco.status = AICopilotRecommendation.STATUS_ACCEPTED
    reco.created_keyword = keyword_obj
    reco.resolved_at = timezone.now()
    reco.save(update_fields=["status", "created_keyword", "resolved_at"])
    return reco


@transaction.atomic
def reject_copilot_recommendation(reco: AICopilotRecommendation) -> AICopilotRecommendation:
    if reco.status == AICopilotRecommendation.STATUS_REJECTED:
        return reco
    reco.status = AICopilotRecommendation.STATUS_REJECTED
    reco.resolved_at = timezone.now()
    reco.save(update_fields=["status", "resolved_at"])
    return reco


@transaction.atomic
def accept_copilot_metadata_variant(variant: AICopilotMetadataVariant) -> AICopilotMetadataVariant:
    if variant.status == AICopilotMetadataVariant.STATUS_ACCEPTED:
        return variant
    AICopilotMetadataVariant.objects.filter(
        app=variant.app,
        status=AICopilotMetadataVariant.STATUS_ACCEPTED,
    ).exclude(id=variant.id).update(
        status=AICopilotMetadataVariant.STATUS_REJECTED,
        resolved_at=timezone.now(),
    )
    variant.status = AICopilotMetadataVariant.STATUS_ACCEPTED
    variant.resolved_at = timezone.now()
    variant.save(update_fields=["status", "resolved_at"])
    return variant


@transaction.atomic
def reject_copilot_metadata_variant(variant: AICopilotMetadataVariant) -> AICopilotMetadataVariant:
    if variant.status == AICopilotMetadataVariant.STATUS_REJECTED:
        return variant
    variant.status = AICopilotMetadataVariant.STATUS_REJECTED
    variant.resolved_at = timezone.now()
    variant.save(update_fields=["status", "resolved_at"])
    return variant
