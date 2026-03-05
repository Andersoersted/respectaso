"""
AI Copilot generation + action handlers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import logging
import re
import time
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
SNAPSHOT_FEATURE_ROWS_LIMIT = 80
SNAPSHOT_EXISTING_KEYWORDS_LIMIT = 120
SNAPSHOT_RETRY_FEATURE_ROWS_LIMIT = 30
SNAPSHOT_RETRY_EXISTING_KEYWORDS_LIMIT = 60
OPENAI_COPILOT_MAX_OUTPUT_TOKENS = 1600
COPILOT_MAX_EVENT_LOG_ENTRIES = 120


class AICopilotError(Exception):
    """Raised when AI Copilot cannot complete safely."""

    def __init__(self, message: str, *, user_error: bool = True):
        super().__init__(message)
        self.user_error = bool(user_error)


class RecommendationOut(BaseModel):
    keyword: str = Field(min_length=2, max_length=80)
    action: str = Field(default="watch")
    rationale: str = Field(default="", max_length=240)
    llm_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class MetadataVariantOut(BaseModel):
    title: str = Field(min_length=1, max_length=60)
    subtitle: str = Field(default="", max_length=60)
    keyword_field: str = Field(default="", max_length=180)
    covered_keywords: list[str] = Field(default_factory=list)
    predicted_impact: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=280)


class CopilotOutput(BaseModel):
    recommendations: list[RecommendationOut] = Field(default_factory=list, max_length=30)
    metadata_variants: list[MetadataVariantOut] = Field(default_factory=list, max_length=6)


@dataclass(slots=True)
class CopilotSettings:
    model: str
    api_key: str
    system_prompt: str
    user_prompt_template: str
    reasoning_effort: str = ""


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


@lru_cache(maxsize=1)
def _openai_exception_types() -> dict[str, type[BaseException]]:
    try:
        import openai
    except Exception:  # pragma: no cover - import guard
        return {}

    names = (
        "APIConnectionError",
        "APITimeoutError",
        "APIStatusError",
        "RateLimitError",
        "AuthenticationError",
        "PermissionDeniedError",
        "BadRequestError",
        "NotFoundError",
    )
    return {name: cls for name in names if isinstance((cls := getattr(openai, name, None)), type)}


def _is_openai_exc(exc: Exception, *names: str) -> bool:
    type_map = _openai_exception_types()
    for name in names:
        exc_type = type_map.get(name)
        if exc_type and isinstance(exc, exc_type):
            return True
    return False


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


def _is_timeout_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return "timed out" in text or "timeout" in text


def _normalize_reasoning_effort(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    allowed = {"", "none", "minimal", "low", "medium", "high", "xhigh"}
    return raw if raw in allowed else ""


def _default_reasoning_effort_for_model(model: str, configured_value: str | None) -> str:
    configured = _normalize_reasoning_effort(configured_value)
    if configured:
        return configured
    if str(model or "").lower().startswith("gpt-5"):
        return "low"
    return ""


def _is_unsupported_reasoning_error(exc: Exception, *, meta: dict[str, Any]) -> bool:
    status_code = meta.get("status_code")
    text = f"{exc} {meta.get('body_excerpt') or ''}".lower()
    return bool(
        status_code == 400
        and "reasoning.effort" in text
        and ("unsupported value" in text or "supported values" in text)
    )


def _is_recoverable_generation_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    if _is_timeout_error(exc):
        return True
    recoverable_markers = (
        "validation error for copilotoutput",
        "json_invalid",
        "eof while parsing",
        "lengthfinishreasonerror",
    )
    return any(marker in text for marker in recoverable_markers)


def _render_prompt(template: str, payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=True)
    prompt = (template or "").replace("{{SNAPSHOT_JSON}}", payload_json)
    if "{{SNAPSHOT_JSON}}" not in (template or ""):
        prompt += f"\n\nCOPILOT SNAPSHOT JSON:\n{payload_json}"
    return prompt


def _compact_feature_rows(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    compact_rows: list[dict[str, Any]] = []
    for row in (rows or [])[: max(1, int(limit))]:
        compact_rows.append(
            {
                "keyword": row.get("keyword"),
                "country": row.get("country"),
                "popularity": row.get("popularity"),
                "difficulty": row.get("difficulty"),
                "opportunity": row.get("opportunity"),
                "app_rank": row.get("app_rank"),
                "prev_app_rank": row.get("prev_app_rank"),
                "score_market_opportunity": row.get("score_market_opportunity"),
                "score_rank_momentum": row.get("score_rank_momentum"),
                "score_business_impact": row.get("score_business_impact"),
                "score_coverage_gap": row.get("score_coverage_gap"),
            }
        )
    return compact_rows


def _compact_snapshot(snapshot: dict[str, Any], *, feature_rows_limit: int, existing_keywords_limit: int) -> dict[str, Any]:
    compact = dict(snapshot or {})
    compact["feature_rows"] = _compact_feature_rows(snapshot.get("feature_rows") or [], limit=feature_rows_limit)
    compact["existing_keywords"] = list((snapshot.get("existing_keywords") or [])[: max(1, int(existing_keywords_limit))])
    compact["snapshot_meta"] = {
        "feature_rows_limit": int(feature_rows_limit),
        "existing_keywords_limit": int(existing_keywords_limit),
    }
    return compact


def _append_run_event(
    run: AICopilotRun | None,
    *,
    stage: str,
    message: str,
    kind: str = "info",
) -> None:
    if not run:
        return
    events = list(run.events_json or [])
    events.append(
        {
            "ts": timezone.now().isoformat(),
            "stage": (stage or "info")[:64],
            "kind": (kind or "info")[:16],
            "message": _truncate_chars(str(message or ""), 500),
        }
    )
    events = events[-COPILOT_MAX_EVENT_LOG_ENTRIES:]
    _update_run_fields(run, events_json=events)


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


def _openai_error_meta(exc: Exception) -> dict[str, Any]:
    status_code = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    body_excerpt = ""
    response = getattr(exc, "response", None)
    response_text = getattr(response, "text", None) if response is not None else None
    if response_text:
        body_excerpt = str(response_text).replace("\n", " ").strip()[:320]
    if not body_excerpt:
        body_obj = getattr(exc, "body", None)
        if body_obj:
            if isinstance(body_obj, (dict, list)):
                body_excerpt = json.dumps(body_obj, ensure_ascii=True)[:320]
            else:
                body_excerpt = str(body_obj).replace("\n", " ").strip()[:320]
    return {
        "exception_type": exc.__class__.__name__,
        "status_code": status_code,
        "request_id": request_id,
        "body_excerpt": body_excerpt,
    }


def _format_openai_exception(exc: Exception) -> tuple[str, bool]:
    msg = str(exc or "").strip() or exc.__class__.__name__
    meta = _openai_error_meta(exc)
    status_code = meta.get("status_code")
    request_id = meta.get("request_id")
    body_excerpt = str(meta.get("body_excerpt") or "")
    text = f"{msg} {body_excerpt}".lower()
    request_hint = f" (request_id={request_id})" if request_id else ""

    if _is_openai_exc(exc, "APITimeoutError"):
        return (
            "OpenAI request timed out. Retry, or increase OPENAI timeout in Config "
            f"(OPENAI_TIMEOUT_SECONDS).{request_hint}",
            False,
        )
    if _is_openai_exc(exc, "APIConnectionError"):
        return (
            "OpenAI connection failed. Check network egress and retry."
            f"{request_hint}",
            False,
        )
    if "validation error for copilotoutput" in text or "json_invalid" in text or "eof while parsing" in text:
        return (
            "OpenAI returned incomplete structured JSON for Copilot output. "
            f"Retry may succeed.{request_hint}",
            False,
        )
    if _is_openai_exc(exc, "RateLimitError") or status_code == 429 or "insufficient_quota" in text or "quota" in text:
        return (
            "OpenAI quota or rate limit was hit. Check OpenAI billing/usage and retry."
            f"{request_hint}",
            True,
        )
    if _is_openai_exc(exc, "AuthenticationError") or status_code == 401:
        return (
            "OpenAI authentication failed. Verify OPENAI_API_KEY in Config."
            f"{request_hint}",
            True,
        )
    if _is_openai_exc(exc, "PermissionDeniedError") or status_code == 403:
        return (
            "OpenAI request was forbidden for this key/model. Check project permissions."
            f"{request_hint}",
            True,
        )
    if _is_openai_exc(exc, "BadRequestError") or status_code == 400:
        return (
            f"OpenAI rejected the request payload: {msg}{request_hint}",
            True,
        )
    if status_code and int(status_code) >= 500:
        return (
            "OpenAI service is temporarily unavailable. Please retry."
            f"{request_hint}",
            False,
        )
    if "timed out" in text or "timeout" in text:
        return (
            "OpenAI request timed out. Retry, or increase OPENAI timeout in Config "
            f"(OPENAI_TIMEOUT_SECONDS).{request_hint}",
            False,
        )
    return (f"OpenAI error: {msg}{request_hint}", True)


def _should_fallback_after_responses_error(exc: Exception, *, meta: dict[str, Any]) -> bool:
    """Only fallback when responses.parse appears unsupported for this route/model."""
    status_code = meta.get("status_code")
    exception_type = str(meta.get("exception_type") or "")
    text = f"{exc} {meta.get('body_excerpt') or ''}".lower()

    if _is_openai_exc(
        exc,
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "AuthenticationError",
        "PermissionDeniedError",
    ):
        return False
    if status_code in {401, 403, 429}:
        return False
    if status_code and int(status_code) >= 500:
        return False
    if "timed out" in text or "timeout" in text:
        return False

    parse_unsupported_markers = (
        "text_format",
        "response_format",
        "path_error",
        "not a valid url",
        "unsupported",
        "unknown parameter",
        "unrecognized",
    )
    has_parse_marker = any(marker in text for marker in parse_unsupported_markers)
    recoverable_parse_markers = (
        "validation error for copilotoutput",
        "json_invalid",
        "eof while parsing",
        "lengthfinishreasonerror",
    )
    has_recoverable_parse_marker = any(marker in text for marker in recoverable_parse_markers)

    if _is_openai_exc(exc, "NotFoundError"):
        return True
    if exception_type in {"ValidationError", "LengthFinishReasonError", "ContentFilterFinishReasonError"}:
        return True
    if has_recoverable_parse_marker:
        return True
    if _is_openai_exc(exc, "BadRequestError") and has_parse_marker:
        return True
    if status_code in {400, 404, 405, 422} and has_parse_marker:
        return True
    return False


def _generate_with_openai(
    *,
    snapshot: dict[str, Any],
    settings_obj: CopilotSettings,
    run: AICopilotRun | None = None,
    run_id: int | None = None,
    attempt: str = "primary",
    timeout_seconds: float | None = None,
) -> CopilotOutput:
    effective_run_id = run_id or (run.id if run else None)
    effective_timeout = float(timeout_seconds or settings.OPENAI_TIMEOUT_SECONDS)
    client = _build_openai_client(settings_obj.api_key)
    request_client = client.with_options(
        timeout=effective_timeout,
        # Keep retries explicit and bounded in app logic to avoid multi-minute cascaded waits.
        max_retries=0,
    )
    user_prompt = _render_prompt(settings_obj.user_prompt_template, snapshot)
    model = settings_obj.model
    reasoning_effort = _default_reasoning_effort_for_model(model, settings_obj.reasoning_effort)
    top_keywords = [
        _clean_keyword(str(row.get("keyword") or ""))
        for row in (snapshot.get("feature_rows") or [])[:8]
        if str(row.get("keyword") or "").strip()
    ]
    payload_summary = {
        "attempt": attempt,
        "model": model,
        "timeout_seconds": round(effective_timeout, 2),
        "reasoning_effort": reasoning_effort or "auto",
        "feature_rows": len(snapshot.get("feature_rows") or []),
        "existing_keywords": len(snapshot.get("existing_keywords") or []),
        "top_keywords": top_keywords,
        "system_prompt_chars": len(settings_obj.system_prompt or ""),
        "user_prompt_chars": len(user_prompt),
    }
    payload_summary_json = json.dumps(payload_summary, ensure_ascii=True)

    responses_parse_kwargs: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": settings_obj.system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "text_format": CopilotOutput,
        "max_output_tokens": OPENAI_COPILOT_MAX_OUTPUT_TOKENS,
    }
    if reasoning_effort:
        responses_parse_kwargs["reasoning"] = {"effort": reasoning_effort}

    logger.warning(
        "Copilot run %s sending payload to OpenAI responses.parse summary=%s",
        effective_run_id or "-",
        payload_summary_json,
    )
    _append_run_event(
        run,
        stage="generating_recommendations",
        message=f"OpenAI request summary: {payload_summary_json}",
    )
    logger.warning(
        "Copilot run %s sending payload to OpenAI responses.parse attempt=%s model=%s reasoning=%s timeout=%ss retries=%s snapshot_rows=%s existing_keywords=%s prompt_chars=%s",
        effective_run_id or "-",
        attempt,
        model,
        reasoning_effort or "auto",
        effective_timeout,
        0,
        len(snapshot.get("feature_rows") or []),
        len(snapshot.get("existing_keywords") or []),
        len(user_prompt),
    )
    responses_started = time.perf_counter()
    try:
        response = request_client.responses.parse(**responses_parse_kwargs)
        parsed = _extract_responses_parse(response)
        if parsed is not None:
            req_id = getattr(response, "_request_id", None)
            output_summary = {
                "route": "responses.parse",
                "attempt": attempt,
                "request_id": req_id or None,
                "recommendations": len(parsed.recommendations),
                "metadata_variants": len(parsed.metadata_variants),
                "recommendation_keywords": [item.keyword for item in parsed.recommendations[:8]],
                "metadata_titles": [item.title for item in parsed.metadata_variants[:5]],
            }
            output_summary_json = json.dumps(output_summary, ensure_ascii=True)
            logger.warning(
                "Copilot run %s responses.parse succeeded in %.2fs summary=%s",
                effective_run_id or "-",
                time.perf_counter() - responses_started,
                output_summary_json,
            )
            _append_run_event(
                run,
                stage="model_output_received",
                message=f"OpenAI response summary: {output_summary_json}",
                kind="success",
            )
            return parsed
        logger.warning(
            "Copilot run %s responses.parse returned no parsed payload in %.2fs attempt=%s; trying chat.completions.parse",
            effective_run_id or "-",
            time.perf_counter() - responses_started,
            attempt,
        )
        _append_run_event(
            run,
            stage="generating_recommendations",
            message=(
                f"responses.parse returned no structured payload in {time.perf_counter() - responses_started:.2f}s; "
                "switching to chat.completions.parse fallback."
            ),
        )
    except Exception as exc:
        meta = _openai_error_meta(exc)
        logger.warning(
            "Copilot run %s responses.parse failed for model %s in %.2fs attempt=%s type=%s status=%s request_id=%s detail=%s body=%s",
            effective_run_id or "-",
            model,
            time.perf_counter() - responses_started,
            attempt,
            meta.get("exception_type"),
            meta.get("status_code"),
            meta.get("request_id") or "-",
            exc,
            meta.get("body_excerpt") or "-",
        )
        _append_run_event(
            run,
            stage="generating_recommendations",
            kind="error",
            message=(
                "responses.parse failed "
                f"(type={meta.get('exception_type')}, status={meta.get('status_code')}, "
                f"request_id={meta.get('request_id') or '-'}): {exc}"
            ),
        )

        if _is_unsupported_reasoning_error(exc, meta=meta) and reasoning_effort:
            fallback_reasoning_effort = "low" if model.lower().startswith("gpt-5") else ""
            if fallback_reasoning_effort == reasoning_effort:
                fallback_reasoning_effort = ""
            if fallback_reasoning_effort != reasoning_effort:
                retry_parse_kwargs = dict(responses_parse_kwargs)
                if fallback_reasoning_effort:
                    retry_parse_kwargs["reasoning"] = {"effort": fallback_reasoning_effort}
                else:
                    retry_parse_kwargs.pop("reasoning", None)
                logger.warning(
                    "Copilot run %s retrying responses.parse with fallback reasoning effort=%s (initial=%s)",
                    effective_run_id or "-",
                    fallback_reasoning_effort or "auto",
                    reasoning_effort,
                )
                _append_run_event(
                    run,
                    stage="retrying_reasoning",
                    message=(
                        "Selected reasoning effort is unsupported for this model. Retrying with "
                        f"reasoning={fallback_reasoning_effort or 'auto'}."
                    ),
                )
                retry_started = time.perf_counter()
                try:
                    retry_response = request_client.responses.parse(**retry_parse_kwargs)
                    retry_parsed = _extract_responses_parse(retry_response)
                    if retry_parsed is not None:
                        retry_req_id = getattr(retry_response, "_request_id", None)
                        output_summary = {
                            "route": "responses.parse",
                            "attempt": f"{attempt}:reasoning_retry",
                            "request_id": retry_req_id or None,
                            "recommendations": len(retry_parsed.recommendations),
                            "metadata_variants": len(retry_parsed.metadata_variants),
                            "recommendation_keywords": [item.keyword for item in retry_parsed.recommendations[:8]],
                            "metadata_titles": [item.title for item in retry_parsed.metadata_variants[:5]],
                        }
                        output_summary_json = json.dumps(output_summary, ensure_ascii=True)
                        logger.warning(
                            "Copilot run %s responses.parse reasoning fallback succeeded in %.2fs summary=%s",
                            effective_run_id or "-",
                            time.perf_counter() - retry_started,
                            output_summary_json,
                        )
                        _append_run_event(
                            run,
                            stage="model_output_received",
                            kind="success",
                            message=f"OpenAI response summary: {output_summary_json}",
                        )
                        return retry_parsed
                    logger.warning(
                        "Copilot run %s responses.parse reasoning fallback returned no parsed payload in %.2fs; using chat.completions.parse fallback",
                        effective_run_id or "-",
                        time.perf_counter() - retry_started,
                    )
                except Exception as retry_exc:
                    retry_meta = _openai_error_meta(retry_exc)
                    logger.warning(
                        "Copilot run %s responses.parse reasoning fallback failed type=%s status=%s request_id=%s detail=%s body=%s",
                        effective_run_id or "-",
                        retry_meta.get("exception_type"),
                        retry_meta.get("status_code"),
                        retry_meta.get("request_id") or "-",
                        retry_exc,
                        retry_meta.get("body_excerpt") or "-",
                    )
                    _append_run_event(
                        run,
                        stage="retrying_reasoning",
                        kind="error",
                        message=(
                            "Reasoning fallback retry failed "
                            f"(type={retry_meta.get('exception_type')}, status={retry_meta.get('status_code')}, "
                            f"request_id={retry_meta.get('request_id') or '-'}): {retry_exc}"
                        ),
                    )

        if not _should_fallback_after_responses_error(exc, meta=meta):
            formatted, user_error = _format_openai_exception(exc)
            raise AICopilotError(f"Copilot generation failed: {formatted}", user_error=user_error) from exc

    # Fallback for models/routes without responses structured parsing.
    logger.warning(
        "Copilot run %s starting chat.completions.parse fallback attempt=%s model=%s",
        effective_run_id or "-",
        attempt,
        model,
    )
    _append_run_event(
        run,
        stage="generating_recommendations",
        message=(
            f"Starting chat.completions.parse fallback (attempt={attempt}, model={model}, "
            f"reasoning={reasoning_effort or 'auto'})."
        ),
    )
    fallback_started = time.perf_counter()
    chat_parse_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": settings_obj.system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": CopilotOutput,
    }

    try:
        completion = request_client.chat.completions.parse(**chat_parse_kwargs)
    except Exception as exc:  # pragma: no cover - runtime fallback path
        meta = _openai_error_meta(exc)
        logger.error(
            "Copilot run %s chat.completions.parse failed for model %s in %.2fs attempt=%s type=%s status=%s request_id=%s detail=%s body=%s",
            effective_run_id or "-",
            model,
            time.perf_counter() - fallback_started,
            attempt,
            meta.get("exception_type"),
            meta.get("status_code"),
            meta.get("request_id") or "-",
            exc,
            meta.get("body_excerpt") or "-",
        )
        _append_run_event(
            run,
            stage="generating_recommendations",
            kind="error",
            message=(
                "chat.completions.parse failed "
                f"(type={meta.get('exception_type')}, status={meta.get('status_code')}, "
                f"request_id={meta.get('request_id') or '-'}): {exc}"
            ),
        )
        formatted, user_error = _format_openai_exception(exc)
        raise AICopilotError(f"Copilot generation failed: {formatted}", user_error=user_error) from exc

    message = completion.choices[0].message if completion.choices else None
    parsed = getattr(message, "parsed", None) if message else None
    if not isinstance(parsed, CopilotOutput):
        refusal = getattr(message, "refusal", "") if message else ""
        raise AICopilotError(refusal or "Copilot model returned no structured output.", user_error=True)
    req_id = getattr(completion, "_request_id", None)
    output_summary = {
        "route": "chat.completions.parse",
        "attempt": attempt,
        "request_id": req_id or None,
        "recommendations": len(parsed.recommendations),
        "metadata_variants": len(parsed.metadata_variants),
        "recommendation_keywords": [item.keyword for item in parsed.recommendations[:8]],
        "metadata_titles": [item.title for item in parsed.metadata_variants[:5]],
    }
    output_summary_json = json.dumps(output_summary, ensure_ascii=True)
    logger.warning(
        "Copilot run %s chat.completions.parse fallback succeeded in %.2fs summary=%s",
        effective_run_id or "-",
        time.perf_counter() - fallback_started,
        output_summary_json,
    )
    _append_run_event(
        run,
        stage="model_output_received",
        kind="success",
        message=f"OpenAI response summary: {output_summary_json}",
    )
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
        Keyword.objects.filter(app=app).values_list("keyword", flat=True).order_by("-created_at")[
            :SNAPSHOT_EXISTING_KEYWORDS_LIMIT
        ]
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
        "feature_rows": _compact_feature_rows(feature_rows, limit=SNAPSHOT_FEATURE_ROWS_LIMIT),
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
    _append_run_event(
        run,
        stage="failed",
        kind="error",
        message=str(error or "Copilot run failed."),
    )


def generate_ai_copilot(
    app: App,
    *,
    country: str,
    settings_obj: CopilotSettings,
    run: AICopilotRun | None = None,
) -> AICopilotRun:
    country = (country or "us").lower()
    if run is None:
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
    else:
        _update_run_fields(
            run,
            status=AICopilotRun.STATUS_RUNNING,
            progress_percent=2,
            progress_stage="preparing",
            progress_detail="Validating inputs and preparing run.",
            model=settings_obj.model,
            country=country,
            input_snapshot_json={},
            feature_summary_json={},
            finished_at=None,
            error="",
        )
    _append_run_event(
        run,
        stage="preparing",
        message=(
            f"Run started for {app.name} ({country.upper()}) with model {settings_obj.model} "
            f"and reasoning {_default_reasoning_effort_for_model(settings_obj.model, settings_obj.reasoning_effort) or 'auto'}."
        ),
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
        _append_run_event(
            run,
            stage="collecting_signals",
            message=(
                f"Collected {len(feature_rows)} feature row(s). "
                f"Latest keyword sample: {', '.join([str(item.get('keyword')) for item in feature_rows[:6]]) or '-'}."
            ),
        )

        snapshot = _build_snapshot(app, country, feature_rows, feature_summary)
        _update_run_fields(
            run,
            input_snapshot_json=snapshot,
            feature_summary_json=feature_summary,
        )
        _append_run_event(
            run,
            stage="collecting_signals",
            message=(
                f"Snapshot prepared with {len(snapshot.get('feature_rows') or [])} feature row(s) and "
                f"{len(snapshot.get('existing_keywords') or [])} existing keyword(s)."
            ),
        )

        _set_run_progress(
            run,
            percent=34,
            stage="generating_recommendations",
            detail=(
                f"Sending snapshot to {settings_obj.model} "
                f"({len(snapshot.get('feature_rows') or [])} feature row(s), "
                f"{len(snapshot.get('existing_keywords') or [])} existing keyword(s))."
            ),
        )
        try:
            output = _generate_with_openai(
                snapshot=snapshot,
                settings_obj=settings_obj,
                run=run,
                run_id=run.id,
                attempt="primary",
                timeout_seconds=settings.OPENAI_TIMEOUT_SECONDS,
            )
        except AICopilotError as exc:
            if _is_recoverable_generation_error(exc):
                compact_snapshot = _compact_snapshot(
                    snapshot,
                    feature_rows_limit=SNAPSHOT_RETRY_FEATURE_ROWS_LIMIT,
                    existing_keywords_limit=SNAPSHOT_RETRY_EXISTING_KEYWORDS_LIMIT,
                )
                if (
                    len(compact_snapshot.get("feature_rows") or []) < len(snapshot.get("feature_rows") or [])
                    or len(compact_snapshot.get("existing_keywords") or []) < len(snapshot.get("existing_keywords") or [])
                ):
                    _update_run_fields(run, input_snapshot_json=compact_snapshot)
                    _set_run_progress(
                        run,
                        percent=40,
                        stage="retrying_compact_prompt",
                        detail=(
                            "Primary model request failed with timeout/parse issue. "
                            "Retrying once with a smaller snapshot."
                        ),
                    )
                    _append_run_event(
                        run,
                        stage="retrying_compact_prompt",
                        message=(
                            f"Retrying with compact snapshot: feature_rows {len(snapshot.get('feature_rows') or [])} -> "
                            f"{len(compact_snapshot.get('feature_rows') or [])}, existing_keywords "
                            f"{len(snapshot.get('existing_keywords') or [])} -> "
                            f"{len(compact_snapshot.get('existing_keywords') or [])}."
                        ),
                    )
                    output = _generate_with_openai(
                        snapshot=compact_snapshot,
                        settings_obj=settings_obj,
                        run=run,
                        run_id=run.id,
                        attempt="compact_retry",
                        timeout_seconds=max(float(settings.OPENAI_TIMEOUT_SECONDS), 60.0),
                    )
                else:
                    raise
            else:
                raise

        logger.warning(
            "Copilot run %s received model output recommendations=%s metadata_variants=%s",
            run.id,
            len(output.recommendations),
            len(output.metadata_variants),
        )
        _set_run_progress(
            run,
            percent=46,
            stage="model_output_received",
            detail=(
                f"Received model output: {len(output.recommendations)} recommendation candidate(s), "
                f"{len(output.metadata_variants)} metadata variant candidate(s)."
            ),
        )
        _append_run_event(
            run,
            stage="model_output_received",
            kind="success",
            message=(
                f"Model output parsed: {len(output.recommendations)} recommendation(s), "
                f"{len(output.metadata_variants)} metadata variant(s)."
            ),
        )

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
                progress_detail=(
                    f"Saved {min(len(rec_objects), 30)} recommendation(s) and "
                    f"{min(len(variant_objects), 3)} metadata variant(s)."
                ),
            )
        _append_run_event(
            run,
            stage="complete",
            kind="success",
            message=(
                f"Saved {min(len(rec_objects), 30)} recommendation(s) and "
                f"{min(len(variant_objects), 3)} metadata variant(s)."
            ),
        )
        return run
    except AICopilotError as exc:
        _mark_run_failed(run, str(exc))
        logger.warning("AI copilot run %s failed for app %s: %s", run.id, app.id, exc)
        raise
    except Exception as exc:
        _mark_run_failed(run, str(exc))
        logger.exception("AI copilot run %s failed unexpectedly for app %s", run.id, app.id)
        raise AICopilotError(str(exc), user_error=False) from exc


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
