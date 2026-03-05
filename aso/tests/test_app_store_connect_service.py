from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from aso.app_store_connect_service import AppStoreConnectService


class AppStoreConnectServicePathTests(TestCase):
    def setUp(self):
        self.service = AppStoreConnectService(
            issuer_id="issuer",
            key_id="key",
            private_key_pem="-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----",
            timeout_seconds=1,
            max_retries=0,
        )

    def test_list_report_instances_uses_analytics_reports_path(self):
        with patch.object(self.service, "_request", return_value={"data": []}) as mock_request:
            self.service.list_report_instances("r39-123")
        mock_request.assert_called_once_with("GET", "/v1/analyticsReports/r39-123/instances")

    def test_list_report_segments_uses_report_instances_path(self):
        with patch.object(self.service, "_request", return_value={"data": []}) as mock_request:
            self.service.list_report_segments("inst-123")
        mock_request.assert_called_once_with("GET", "/v1/analyticsReportInstances/inst-123/segments")

    def test_get_report_segment_uses_correct_segment_path(self):
        with patch.object(self.service, "_request", return_value={"data": {}}) as mock_request:
            self.service.get_report_segment("seg-123")
        mock_request.assert_called_once_with("GET", "/v1/analyticsReportSegments/seg-123")

    def test_fetch_metric_rows_walks_reports_instances_segments(self):
        parsed_rows = [{"date": "2026-03-05", "country": "us"}]
        with (
            patch.object(self.service, "create_report_request", return_value="req-1") as mock_create,
            patch.object(self.service, "list_reports_for_request", return_value=[{"id": "r39-1"}]) as mock_reports,
            patch.object(self.service, "list_report_instances", return_value=[{"id": "inst-1"}]) as mock_instances,
            patch.object(
                self.service,
                "list_report_segments",
                return_value=[{"id": "seg-1", "attributes": {"url": "https://download/seg-1"}}],
            ) as mock_segments,
            patch.object(self.service, "download_segment_payload", return_value={"rows": []}) as mock_download,
            patch.object(self.service, "parse_metric_rows", return_value=parsed_rows) as mock_parse,
            patch.object(self.service, "get_report_segment") as mock_get_segment,
        ):
            rows = self.service.fetch_metric_rows("123456")

        self.assertEqual(rows, parsed_rows)
        mock_create.assert_called_once_with("123456")
        mock_reports.assert_called_once_with("req-1")
        mock_instances.assert_called_once_with("r39-1")
        mock_segments.assert_called_once_with("inst-1")
        mock_download.assert_called_once_with("https://download/seg-1")
        mock_parse.assert_called_once()
        mock_get_segment.assert_not_called()

    def test_fetch_metric_rows_falls_back_to_segment_lookup_when_url_missing(self):
        with (
            patch.object(self.service, "create_report_request", return_value="req-1"),
            patch.object(self.service, "list_reports_for_request", return_value=[{"id": "r39-1"}]),
            patch.object(self.service, "list_report_instances", return_value=[{"id": "inst-1"}]),
            patch.object(
                self.service,
                "list_report_segments",
                return_value=[{"id": "seg-1", "attributes": {}}],
            ),
            patch.object(
                self.service,
                "get_report_segment",
                return_value={"attributes": {"url": "https://download/seg-1"}},
            ) as mock_get_segment,
            patch.object(self.service, "download_segment_payload", return_value={"rows": []}) as mock_download,
            patch.object(self.service, "parse_metric_rows", return_value=[]),
        ):
            rows = self.service.fetch_metric_rows("123456")

        self.assertEqual(rows, [])
        mock_get_segment.assert_called_once_with("seg-1")
        mock_download.assert_called_once_with("https://download/seg-1")
