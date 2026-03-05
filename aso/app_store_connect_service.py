"""
App Store Connect analytics synchronization service.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import logging
import time
from typing import Any

from django.conf import settings
from django.db import transaction
import requests

from .models import ASCMetricDaily, App

logger = logging.getLogger(__name__)


class ASCError(Exception):
    """Base error for App Store Connect integration."""


class ASCAuthError(ASCError):
    """Authentication/credential failure."""


class ASCAPIError(ASCError):
    """Upstream API failure."""


class AppStoreConnectService:
    BASE_URL = "https://api.appstoreconnect.apple.com"

    def __init__(
        self,
        *,
        issuer_id: str,
        key_id: str,
        private_key_pem: str,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ):
        self.issuer_id = (issuer_id or "").strip()
        self.key_id = (key_id or "").strip()
        self.private_key_pem = (private_key_pem or "").strip()
        self.timeout_seconds = (
            settings.ASC_TIMEOUT_SECONDS if timeout_seconds is None else float(timeout_seconds)
        )
        self.max_retries = settings.ASC_MAX_RETRIES if max_retries is None else int(max_retries)
        self.session = requests.Session()

        if not self.issuer_id or not self.key_id or not self.private_key_pem:
            raise ASCAuthError("App Store Connect credentials are incomplete.")

    def _build_token(self) -> str:
        try:
            import jwt
        except Exception as exc:  # pragma: no cover - import guard
            raise ASCAuthError("PyJWT is required for App Store Connect integration.") from exc

        exp = datetime.now(timezone.utc) + timedelta(minutes=settings.ASC_JWT_TTL_MINUTES)
        try:
            token = jwt.encode(
                {
                    "iss": self.issuer_id,
                    "aud": "appstoreconnect-v1",
                    "exp": exp,
                },
                self.private_key_pem,
                algorithm="ES256",
                headers={
                    "kid": self.key_id,
                    "typ": "JWT",
                },
            )
        except Exception as exc:  # pragma: no cover - defensive runtime path
            raise ASCAuthError(f"Failed to sign App Store Connect JWT: {exc}") from exc

        if isinstance(token, bytes):
            token = token.decode("utf-8")
        return token

    def _headers(self) -> dict[str, str]:
        token = self._build_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        path_or_url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        absolute_url: bool = False,
    ) -> dict[str, Any]:
        url = path_or_url if absolute_url else f"{self.BASE_URL}{path_or_url}"
        attempts = max(1, self.max_retries + 1)
        last_exc: Exception | None = None
        last_status: int | None = None
        last_body_excerpt = ""

        for attempt in range(1, attempts + 1):
            try:
                response = self.session.request(
                    method.upper(),
                    url,
                    params=params,
                    json=json_body,
                    headers=self._headers(),
                    timeout=self.timeout_seconds,
                )
                last_status = response.status_code
                last_body_excerpt = (response.text or "").strip().replace("\n", " ")[:400]
                if response.status_code in {401, 403}:
                    msg = "App Store Connect authentication failed."
                    if last_body_excerpt:
                        msg = f"{msg} Details: {last_body_excerpt}"
                    raise ASCAuthError(msg)
                if response.status_code >= 500 and attempt < attempts:
                    logger.warning(
                        "ASC %s %s failed with %s (attempt %s/%s). Retrying.",
                        method.upper(),
                        url,
                        response.status_code,
                        attempt,
                        attempts,
                    )
                    time.sleep(min(2.0, attempt * 0.5))
                    continue
                if response.status_code == 429 and attempt < attempts:
                    logger.warning(
                        "ASC %s %s was rate-limited (429) (attempt %s/%s). Retrying.",
                        method.upper(),
                        url,
                        attempt,
                        attempts,
                    )
                    time.sleep(min(5.0, attempt * 1.0))
                    continue
                response.raise_for_status()
                if not response.content:
                    return {}
                return response.json()
            except ASCAuthError:
                raise
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= attempts:
                    break
                time.sleep(min(2.0, attempt * 0.5))
            except ValueError as exc:
                raise ASCAPIError(f"App Store Connect returned non-JSON data for {url}.") from exc

        detail = f"status={last_status}" if last_status is not None else "status=unknown"
        if last_body_excerpt:
            detail = f"{detail}, body={last_body_excerpt}"
        raise ASCAPIError(f"App Store Connect request failed for {url}: {last_exc} ({detail})")

    def create_report_request(self, asc_app_id: str) -> str:
        payload = {
            "data": {
                "type": "analyticsReportRequests",
                "attributes": {"accessType": "ONGOING"},
                "relationships": {
                    "app": {"data": {"type": "apps", "id": str(asc_app_id)}},
                },
            }
        }
        try:
            response = self._request("POST", "/v1/analyticsReportRequests", json_body=payload)
            request_id = (((response or {}).get("data") or {}).get("id") or "").strip()
            if not request_id:
                raise ASCAPIError("App Store Connect did not return a report request id.")
            return request_id
        except ASCAPIError as exc:
            message = str(exc)
            if "status=409" not in message:
                raise
            existing_id = self.find_existing_report_request_id(
                asc_app_id,
                preferred_access_type="ONGOING",
            )
            if existing_id:
                logger.warning(
                    "ASC report request already exists for app %s; reusing request %s.",
                    asc_app_id,
                    existing_id,
                )
                return existing_id
            raise

    def list_reports_for_request(self, request_id: str) -> list[dict[str, Any]]:
        response = self._request("GET", f"/v1/analyticsReportRequests/{request_id}/reports")
        return list((response or {}).get("data") or [])

    def list_report_requests_for_app(self, asc_app_id: str) -> list[dict[str, Any]]:
        response = self._request(
            "GET",
            f"/v1/apps/{asc_app_id}/analyticsReportRequests",
            params={
                "limit": 200,
                "fields[analyticsReportRequests]": "accessType",
            },
        )
        return list((response or {}).get("data") or [])

    def find_existing_report_request_id(self, asc_app_id: str, *, preferred_access_type: str = "ONGOING") -> str:
        requests_data = self.list_report_requests_for_app(asc_app_id)
        preferred = []
        fallback = []
        for row in requests_data:
            if not isinstance(row, dict):
                continue
            request_id = str(row.get("id") or "").strip()
            if not request_id:
                continue
            attrs = row.get("attributes") if isinstance(row.get("attributes"), dict) else {}
            access_type = str((attrs or {}).get("accessType") or "").strip().upper()
            if access_type == preferred_access_type.upper():
                preferred.append(request_id)
            else:
                fallback.append(request_id)
        if preferred:
            return preferred[0]
        if fallback:
            return fallback[0]
        return ""

    def list_report_instances(self, report_id: str) -> list[dict[str, Any]]:
        response = self._request("GET", f"/v1/analyticsReports/{report_id}/instances")
        return list((response or {}).get("data") or [])

    def list_report_segments(self, instance_id: str) -> list[dict[str, Any]]:
        response = self._request("GET", f"/v1/analyticsReportInstances/{instance_id}/segments")
        return list((response or {}).get("data") or [])

    def get_report_segment(self, segment_id: str) -> dict[str, Any]:
        response = self._request("GET", f"/v1/analyticsReportSegments/{segment_id}")
        return (response or {}).get("data") or {}

    def download_segment_payload(self, url: str) -> dict[str, Any]:
        return self._request("GET", url, absolute_url=True)

    def _parse_date(self, raw: Any) -> date | None:
        text = str(raw or "").strip()
        if not text:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            return None

    def _to_int(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _to_decimal(self, value: Any) -> Decimal:
        if value is None or value == "":
            return Decimal("0.00")
        try:
            return Decimal(str(value))
        except (TypeError, ValueError, InvalidOperation):
            return Decimal("0.00")

    def _extract_rows(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        candidates = []
        for key in ("rows", "data", "results", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates.extend([item for item in value if isinstance(item, dict)])
        if not candidates and isinstance(payload.get("data"), dict):
            nested = payload["data"]
            for key in ("rows", "data", "results", "items"):
                value = nested.get(key)
                if isinstance(value, list):
                    candidates.extend([item for item in value if isinstance(item, dict)])
        return candidates

    def parse_metric_rows(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for row in self._extract_rows(payload):
            attrs = row.get("attributes") if isinstance(row.get("attributes"), dict) else row
            if not isinstance(attrs, dict):
                continue
            parsed_date = self._parse_date(
                attrs.get("date")
                or attrs.get("day")
                or attrs.get("reportDate")
                or attrs.get("eventDate")
            )
            country = (
                attrs.get("country")
                or attrs.get("countryCode")
                or attrs.get("territory")
                or attrs.get("regionCode")
                or "us"
            )
            country = str(country).strip().lower()[:5] or "us"
            if parsed_date is None:
                continue

            impressions = self._to_int(
                attrs.get("impressions")
                or attrs.get("impressionCount")
            )
            page_views = self._to_int(
                attrs.get("productPageViews")
                or attrs.get("product_page_views")
                or attrs.get("pageViews")
            )
            app_units = self._to_int(
                attrs.get("appUnits")
                or attrs.get("downloads")
                or attrs.get("installs")
            )
            conversion_rate = attrs.get("conversionRate") or attrs.get("conversion_rate")
            try:
                conversion_rate = float(conversion_rate) if conversion_rate is not None else None
            except (TypeError, ValueError):
                conversion_rate = None
            proceeds = self._to_decimal(attrs.get("proceeds") or attrs.get("revenue"))

            if (
                impressions is None
                and page_views is None
                and app_units is None
                and conversion_rate is None
                and proceeds == Decimal("0.00")
            ):
                continue

            rows.append(
                {
                    "date": parsed_date,
                    "country": country,
                    "impressions": impressions,
                    "product_page_views": page_views,
                    "app_units": app_units,
                    "conversion_rate": conversion_rate,
                    "proceeds": proceeds,
                    "raw_json": attrs,
                }
            )
        return rows

    def fetch_metric_rows(self, asc_app_id: str) -> list[dict[str, Any]]:
        request_id = self.create_report_request(asc_app_id)
        reports = self.list_reports_for_request(request_id)

        all_rows: list[dict[str, Any]] = []
        for report in reports:
            report_id = str(report.get("id") or "").strip()
            if not report_id:
                continue
            instances = self.list_report_instances(report_id)
            if not instances:
                logger.debug("ASC report %s has no instances.", report_id)
                continue

            for instance in instances:
                instance_id = str(instance.get("id") or "").strip()
                if not instance_id:
                    continue
                segments = self.list_report_segments(instance_id)
                if not segments:
                    logger.debug("ASC report instance %s has no segments.", instance_id)
                    continue

                for segment in segments:
                    attrs = (segment.get("attributes") or {}) if isinstance(segment, dict) else {}
                    segment_url = (
                        attrs.get("url")
                        or attrs.get("reportSegmentURL")
                        or attrs.get("downloadURL")
                        or ""
                    )
                    segment_url = str(segment_url).strip()
                    if not segment_url:
                        segment_id = str(segment.get("id") or "").strip()
                        if not segment_id:
                            continue
                        segment_data = self.get_report_segment(segment_id)
                        segment_attrs = (
                            (segment_data.get("attributes") or {})
                            if isinstance(segment_data, dict)
                            else {}
                        )
                        segment_url = (
                            segment_attrs.get("url")
                            or segment_attrs.get("reportSegmentURL")
                            or segment_attrs.get("downloadURL")
                            or ""
                        )
                        segment_url = str(segment_url).strip()
                    if not segment_url:
                        continue
                    payload = self.download_segment_payload(segment_url)
                    all_rows.extend(self.parse_metric_rows(payload))
        return all_rows

    def upsert_metric_rows(self, app: App, rows: list[dict[str, Any]], *, days_back: int) -> int:
        cutoff = date.today() - timedelta(days=max(1, int(days_back)))
        touched = 0
        with transaction.atomic():
            for row in rows:
                row_date = row.get("date")
                if row_date is None or row_date < cutoff:
                    continue
                _, _created = ASCMetricDaily.objects.update_or_create(
                    app=app,
                    date=row_date,
                    country=row.get("country", "us"),
                    defaults={
                        "impressions": row.get("impressions"),
                        "product_page_views": row.get("product_page_views"),
                        "app_units": row.get("app_units"),
                        "conversion_rate": row.get("conversion_rate"),
                        "proceeds": row.get("proceeds", Decimal("0.00")),
                        "raw_json": row.get("raw_json") or {},
                    },
                )
                touched += 1
        return touched

    def sync_app_metrics(self, app: App, *, days_back: int, asc_app_id: str | None = None) -> int:
        resolved_asc_app_id = str(asc_app_id or app.asc_app_id or app.track_id or "").strip()
        if not resolved_asc_app_id:
            raise ASCError("This app is missing `asc_app_id` (and has no track_id fallback).")
        rows = self.fetch_metric_rows(resolved_asc_app_id)
        return self.upsert_metric_rows(app, rows, days_back=days_back)
