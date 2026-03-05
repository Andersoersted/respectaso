from __future__ import annotations

import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import ANY, patch

from django.core.management import call_command
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from aso.ai_service import AISuggestionError, accept_ai_suggestion, generate_ai_suggestions
from aso.copilot_ai_service import (
    AICopilotError,
    CopilotOutput,
    CopilotSettings,
    MetadataVariantOut,
    RecommendationOut,
    _generate_with_openai,
    generate_ai_copilot,
)
from aso.models import (
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
from aso.refresh_service import cleanup_old_results


class HistoryModelTests(TestCase):
    def test_create_snapshot_appends_and_sets_source(self):
        app = App.objects.create(name="My App", track_id=123456789)
        kw = Keyword.objects.create(keyword="aso test", app=app)

        a = SearchResult.create_snapshot(
            keyword=kw,
            country="us",
            source=SearchResult.SOURCE_MANUAL_REFRESH,
            popularity_score=40,
            difficulty_score=50,
            difficulty_breakdown={},
            competitors_data=[],
        )
        b = SearchResult.create_snapshot(
            keyword=kw,
            country="us",
            source=SearchResult.SOURCE_BULK_REFRESH,
            popularity_score=42,
            difficulty_score=49,
            difficulty_breakdown={},
            competitors_data=[],
        )

        self.assertNotEqual(a.id, b.id)
        self.assertEqual(
            SearchResult.objects.filter(keyword=kw, country="us").count(),
            2,
        )
        self.assertEqual(b.source, SearchResult.SOURCE_BULK_REFRESH)

    @override_settings(RESULT_RETENTION_DAYS=1)
    def test_cleanup_old_results_respects_retention_setting(self):
        app = App.objects.create(name="Retention App", track_id=999)
        kw = Keyword.objects.create(keyword="retention", app=app)
        old_row = SearchResult.create_snapshot(
            keyword=kw,
            country="us",
            source=SearchResult.SOURCE_MANUAL_SEARCH,
            popularity_score=10,
            difficulty_score=60,
            difficulty_breakdown={},
            competitors_data=[],
        )
        recent_row = SearchResult.create_snapshot(
            keyword=kw,
            country="gb",
            source=SearchResult.SOURCE_MANUAL_SEARCH,
            popularity_score=11,
            difficulty_score=59,
            difficulty_breakdown={},
            competitors_data=[],
        )

        cutoff_dt = timezone.now() - timedelta(days=5)
        SearchResult.objects.filter(id=old_row.id).update(searched_at=cutoff_dt)
        deleted = cleanup_old_results()

        self.assertGreaterEqual(deleted, 1)
        self.assertFalse(SearchResult.objects.filter(id=old_row.id).exists())
        self.assertTrue(SearchResult.objects.filter(id=recent_row.id).exists())


class RefreshCommandTests(TestCase):
    def _fake_refresh_pair(self, **kwargs):
        keyword_obj = kwargs["keyword_obj"]
        country = kwargs["country"]
        source = kwargs["source"]
        return SearchResult.create_snapshot(
            keyword=keyword_obj,
            country=country,
            source=source,
            popularity_score=30,
            difficulty_score=40,
            difficulty_breakdown={},
            competitors_data=[],
            app_rank=None,
        )

    @patch("aso.refresh_service.refresh_pair")
    def test_refresh_command_creates_refreshrun_and_is_idempotent_daily(self, mock_refresh_pair):
        mock_refresh_pair.side_effect = self._fake_refresh_pair

        app = App.objects.create(name="Command App", track_id=111)
        kw = Keyword.objects.create(keyword="daily keyword", app=app)

        SearchResult.create_snapshot(
            keyword=kw,
            country="us",
            source=SearchResult.SOURCE_MANUAL_SEARCH,
            popularity_score=20,
            difficulty_score=50,
            difficulty_breakdown={},
            competitors_data=[],
        )

        call_command("refresh_tracked_keywords", trigger="cron")
        first = RefreshRun.objects.first()
        self.assertIsNotNone(first)
        self.assertEqual(first.trigger, RefreshRun.TRIGGER_CRON)
        self.assertEqual(first.total, 1)
        self.assertEqual(first.status, RefreshRun.STATUS_SUCCESS)
        self.assertEqual(
            SearchResult.objects.filter(source=SearchResult.SOURCE_DAILY_REFRESH).count(),
            1,
        )

        call_command("refresh_tracked_keywords", trigger="cron")
        latest = RefreshRun.objects.first()
        self.assertEqual(latest.total, 0)
        self.assertEqual(
            SearchResult.objects.filter(source=SearchResult.SOURCE_DAILY_REFRESH).count(),
            1,
        )


class DashboardRankVisibilityTests(TestCase):
    def _snapshot(self, keyword: Keyword, *, country: str = "us", app_rank: int | None = None):
        return SearchResult.create_snapshot(
            keyword=keyword,
            country=country,
            source=SearchResult.SOURCE_MANUAL_SEARCH,
            popularity_score=25,
            difficulty_score=45,
            difficulty_breakdown={},
            competitors_data=[],
            app_rank=app_rank,
        )

    def test_show_rank_true_for_all_apps_when_trackable_rows_exist(self):
        app = App.objects.create(name="Trackable", track_id=123456)
        kw = Keyword.objects.create(keyword="trackable term", app=app)
        self._snapshot(kw, app_rank=None)

        response = self.client.get(reverse("aso:dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context["show_rank"])

    def test_show_rank_false_for_all_apps_without_trackable_or_rank(self):
        kw = Keyword.objects.create(keyword="manual term")
        self._snapshot(kw, app_rank=None)

        response = self.client.get(reverse("aso:dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context["show_rank"])

    def test_show_rank_true_for_filtered_app_when_rank_data_exists(self):
        app = App.objects.create(name="Manual App")
        kw = Keyword.objects.create(keyword="ranked term", app=app)
        self._snapshot(kw, app_rank=23)

        response = self.client.get(reverse("aso:dashboard"), {"app": app.id})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context["show_rank"])


class BulkRefreshAsyncEndpointTests(TestCase):
    def _snapshot(self, keyword: Keyword, *, country: str):
        return SearchResult.create_snapshot(
            keyword=keyword,
            country=country,
            source=SearchResult.SOURCE_MANUAL_SEARCH,
            popularity_score=33,
            difficulty_score=44,
            difficulty_breakdown={},
            competitors_data=[],
            app_rank=None,
        )

    @patch("aso.views.threading.Thread")
    def test_bulk_refresh_starts_async_with_all_apps_scope(self, mock_thread):
        app_a = App.objects.create(name="App A", track_id=111)
        app_b = App.objects.create(name="App B", track_id=222)
        kw_a = Keyword.objects.create(keyword="alpha", app=app_a)
        kw_b = Keyword.objects.create(keyword="beta", app=app_b)
        kw_c = Keyword.objects.create(keyword="gamma")
        self._snapshot(kw_a, country="us")
        self._snapshot(kw_a, country="gb")
        self._snapshot(kw_b, country="us")

        response = self.client.post(
            reverse("aso:keywords_bulk_refresh"),
            data='{"app_id": null}',
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertTrue(payload["started"])
        self.assertEqual(payload["total_pairs"], 4)
        self.assertIn("run_id", payload)

        run = RefreshRun.objects.get(pk=payload["run_id"])
        self.assertEqual(run.status, RefreshRun.STATUS_RUNNING)
        self.assertEqual(run.total, 4)

        mock_thread.assert_called_once()
        thread_kwargs = mock_thread.call_args.kwargs["kwargs"]
        self.assertEqual(thread_kwargs["run_id"], run.id)
        self.assertEqual(
            set(thread_kwargs["pairs"]),
            {
                (kw_a.id, "us"),
                (kw_a.id, "gb"),
                (kw_b.id, "us"),
                (kw_c.id, "us"),
            },
        )

    @patch("aso.views.threading.Thread")
    def test_bulk_refresh_selected_app_scope_only(self, mock_thread):
        app_a = App.objects.create(name="App A", track_id=111)
        app_b = App.objects.create(name="App B", track_id=222)
        kw_a = Keyword.objects.create(keyword="alpha", app=app_a)
        kw_b = Keyword.objects.create(keyword="beta", app=app_b)
        self._snapshot(kw_a, country="us")
        self._snapshot(kw_a, country="gb")
        self._snapshot(kw_b, country="us")

        response = self.client.post(
            reverse("aso:keywords_bulk_refresh"),
            data='{"app_id": %d}' % app_a.id,
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["started"])
        self.assertEqual(payload["total_pairs"], 2)

        thread_kwargs = mock_thread.call_args.kwargs["kwargs"]
        self.assertEqual(set(thread_kwargs["pairs"]), {(kw_a.id, "us"), (kw_a.id, "gb")})

    @patch("aso.views.threading.Thread")
    def test_bulk_refresh_returns_not_started_when_no_keywords(self, mock_thread):
        response = self.client.post(
            reverse("aso:keywords_bulk_refresh"),
            data='{"app_id": 999999}',
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertFalse(payload["started"])
        self.assertEqual(payload["total_pairs"], 0)
        mock_thread.assert_not_called()

    def test_bulk_refresh_returns_409_when_non_stale_run_exists(self):
        RefreshRun.objects.create(
            trigger=RefreshRun.TRIGGER_MANUAL,
            status=RefreshRun.STATUS_RUNNING,
            total=1,
            completed=0,
            current_keyword="alpha (US)",
        )
        app = App.objects.create(name="App A", track_id=111)
        kw = Keyword.objects.create(keyword="alpha", app=app)
        self._snapshot(kw, country="us")

        response = self.client.post(
            reverse("aso:keywords_bulk_refresh"),
            data='{"app_id": null}',
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 409)
        payload = response.json()
        self.assertFalse(payload["success"])

    @patch("aso.views.threading.Thread")
    def test_bulk_refresh_marks_stale_run_failed_and_starts_new(self, mock_thread):
        stale = RefreshRun.objects.create(
            trigger=RefreshRun.TRIGGER_MANUAL,
            status=RefreshRun.STATUS_RUNNING,
            total=2,
            completed=1,
            current_keyword="stale (US)",
        )
        stale_time = timezone.now() - timedelta(hours=5)
        RefreshRun.objects.filter(pk=stale.pk).update(started_at=stale_time)

        app = App.objects.create(name="App A", track_id=111)
        kw = Keyword.objects.create(keyword="alpha", app=app)
        self._snapshot(kw, country="us")

        response = self.client.post(
            reverse("aso:keywords_bulk_refresh"),
            data='{"app_id": null}',
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["started"])

        stale.refresh_from_db()
        self.assertEqual(stale.status, RefreshRun.STATUS_FAILED)
        self.assertIn("Marked stale", stale.error)


class AISuggestionServiceTests(TestCase):
    def test_generate_ai_suggestions_requires_track_id(self):
        app = App.objects.create(name="No Track")
        with self.assertRaises(AISuggestionError):
            generate_ai_suggestions(app)

    @patch("aso.ai_service._evaluate_keyword")
    @patch("aso.ai_service._generate_candidates")
    def test_generate_ai_suggestions_creates_drafts(self, mock_generate_candidates, mock_evaluate):
        app = App.objects.create(name="AI App", track_id=222, bundle_id="com.example.ai")
        Keyword.objects.create(keyword="existing keyword", app=app)

        mock_generate_candidates.return_value = [
            {
                "keyword": "existing keyword",
                "intent": "duplicate",
                "rationale": "duplicate should be removed",
                "relevance_confidence": 0.9,
            },
            {
                "keyword": "new keyword one",
                "intent": "discoverability",
                "rationale": "good fit",
                "relevance_confidence": 0.8,
            },
            {
                "keyword": "new keyword one",
                "intent": "duplicate candidate",
                "rationale": "duplicate",
                "relevance_confidence": 0.7,
            },
        ]
        mock_evaluate.return_value = {
            "countries": [
                {
                    "country": "us",
                    "popularity": 50,
                    "difficulty": 40,
                    "opportunity": 30,
                    "app_rank": 9,
                    "difficulty_breakdown": {},
                    "competitors_data": [],
                }
            ],
            "score_market": 0.3,
        }

        run = generate_ai_suggestions(app)
        self.assertEqual(run.status, AISuggestionRun.STATUS_SUCCESS)
        suggestions = list(run.suggestions.all())
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0].keyword, "new keyword one")
        self.assertEqual(suggestions[0].status, AISuggestion.STATUS_DRAFT)

    def test_accept_ai_suggestion_creates_keyword_and_snapshot(self):
        app = App.objects.create(name="Accept App", track_id=333)
        run = AISuggestionRun.objects.create(
            app=app,
            status=AISuggestionRun.STATUS_SUCCESS,
            model="gpt-4.1-mini",
            countries_json=["us"],
            input_snapshot_json={},
        )
        suggestion = AISuggestion.objects.create(
            run=run,
            app=app,
            keyword="accept me",
            rationale="reason",
            confidence=0.7,
            market_metrics_json={
                "countries": [
                    {
                        "country": "us",
                        "popularity": 45,
                        "difficulty": 55,
                        "opportunity": 20,
                        "app_rank": 12,
                        "difficulty_breakdown": {"foo": 1},
                        "competitors_data": [],
                    }
                ]
            },
        )

        updated = accept_ai_suggestion(suggestion)
        self.assertEqual(updated.status, AISuggestion.STATUS_ACCEPTED)
        self.assertIsNotNone(updated.created_keyword_id)
        self.assertTrue(
            SearchResult.objects.filter(
                keyword=updated.created_keyword,
                source=SearchResult.SOURCE_AI_SUGGESTION,
                country="us",
            ).exists()
        )


class AISuggestionEndpointTests(TestCase):
    def setUp(self):
        self.app = App.objects.create(name="Endpoint App", track_id=123456)
        self.run = AISuggestionRun.objects.create(
            app=self.app,
            status=AISuggestionRun.STATUS_SUCCESS,
            model="gpt-4.1-mini",
            countries_json=["us"],
            input_snapshot_json={},
        )
        self.suggestion = AISuggestion.objects.create(
            run=self.run,
            app=self.app,
            keyword="endpoint keyword",
            rationale="endpoint rationale",
            confidence=0.6,
            market_metrics_json={"countries": []},
        )

    @patch("aso.views.generate_ai_suggestions")
    def test_generate_endpoint_returns_run_and_suggestions(self, mock_generate):
        mock_generate.return_value = self.run
        url = reverse("aso:ai_suggestions_generate")
        response = self.client.post(
            url,
            data='{"app_id": %d, "model": "gpt-4.1-mini"}' % self.app.id,
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        mock_generate.assert_called_once_with(
            self.app,
            model="gpt-4.1-mini",
            api_key=ANY,
            countries=["us"],
            system_prompt=ANY,
            user_prompt_template=ANY,
            enable_online_context=ANY,
            online_top_apps_per_country=ANY,
            history_rows_max=ANY,
        )
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["run"]["id"], self.run.id)

    @override_settings(OPENAI_AVAILABLE_MODELS=["gpt-4.1-mini"])
    def test_generate_endpoint_rejects_model_not_in_allow_list(self):
        url = reverse("aso:ai_suggestions_generate")
        response = self.client.post(
            url,
            data='{"app_id": %d, "model": "gpt-4.1"}' % self.app.id,
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_generate_endpoint_rejects_invalid_country(self):
        url = reverse("aso:ai_suggestions_generate")
        response = self.client.post(
            url,
            data='{"app_id": %d, "model": "gpt-4.1-mini", "country": "xx-invalid"}' % self.app.id,
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_list_endpoint_returns_suggestions(self):
        url = reverse("aso:ai_suggestions")
        response = self.client.get(url, {"app_id": self.app.id})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["app"]["id"], self.app.id)
        self.assertGreaterEqual(len(payload["suggestions"]), 1)

    def test_suggestions_page_renders_without_app_id(self):
        response = self.client.get(reverse("aso:ai_suggestions"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "AI Copilot")

    def test_list_endpoint_rejects_invalid_app_id(self):
        response = self.client.get(reverse("aso:ai_suggestions"), {"app_id": "not-an-int"})
        self.assertEqual(response.status_code, 400)

    def test_accept_and_reject_endpoints(self):
        accept_url = reverse("aso:ai_suggestion_accept", args=[self.suggestion.id])
        reject_url = reverse("aso:ai_suggestion_reject", args=[self.suggestion.id])

        accept_resp = self.client.post(accept_url)
        self.assertEqual(accept_resp.status_code, 200)
        self.suggestion.refresh_from_db()
        self.assertEqual(self.suggestion.status, AISuggestion.STATUS_ACCEPTED)

        # New draft for reject flow
        draft = AISuggestion.objects.create(
            run=self.run,
            app=self.app,
            keyword="reject me",
            rationale="reject",
            confidence=0.4,
            market_metrics_json={"countries": []},
        )
        reject_resp = self.client.post(reverse("aso:ai_suggestion_reject", args=[draft.id]))
        self.assertEqual(reject_resp.status_code, 200)
        draft.refresh_from_db()
        self.assertEqual(draft.status, AISuggestion.STATUS_REJECTED)


class RuntimeConfigViewTests(TestCase):
    def test_config_page_renders(self):
        response = self.client.get(reverse("aso:config"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Config")

    def test_config_page_updates_runtime_settings(self):
        response = self.client.post(
            reverse("aso:config"),
            data={
                "openai_api_key": "sk-test-123",
                "openai_default_model": "gpt-4.1-mini",
                "openai_available_models": "gpt-4.1-mini,gpt-4.1",
            },
        )
        self.assertEqual(response.status_code, 200)
        cfg = RuntimeConfig.get_solo()
        self.assertEqual(cfg.openai_api_key, "sk-test-123")
        self.assertEqual(cfg.openai_default_model, "gpt-4.1-mini")
        self.assertEqual(cfg.openai_available_models, "gpt-4.1-mini,gpt-4.1")

    def test_config_page_keeps_existing_api_key_when_new_key_blank(self):
        cfg = RuntimeConfig.get_solo()
        cfg.openai_api_key = "sk-existing-123"
        cfg.save()

        response = self.client.post(
            reverse("aso:config"),
            data={
                "openai_api_key": "",
            },
        )
        self.assertEqual(response.status_code, 200)
        cfg.refresh_from_db()
        self.assertEqual(cfg.openai_api_key, "sk-existing-123")

    def test_config_page_prefers_new_api_key_even_when_clear_checked(self):
        cfg = RuntimeConfig.get_solo()
        cfg.openai_api_key = "sk-old-123"
        cfg.save()

        response = self.client.post(
            reverse("aso:config"),
            data={
                "openai_api_key": "sk-new-456",
                "clear_openai_api_key": "on",
            },
        )
        self.assertEqual(response.status_code, 200)
        cfg.refresh_from_db()
        self.assertEqual(cfg.openai_api_key, "sk-new-456")

    @override_settings(OPENAI_API_KEY="sk-env-123", OPENAI_MODEL="gpt-5-mini", OPENAI_AVAILABLE_MODELS=["gpt-5-mini"])
    @patch("openai.OpenAI")
    def test_config_openai_test_endpoint_success(self, mock_openai):
        response_obj = type("Response", (), {"output_text": "ok", "_request_id": "req_test_123"})()
        client = mock_openai.return_value
        client.with_options.return_value = client
        client.responses.create.return_value = response_obj

        response = self.client.post(
            reverse("aso:config_openai_test"),
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["model"], "gpt-5-mini")
        self.assertEqual(payload["request_id"], "req_test_123")
        client.responses.create.assert_called_once()

    @override_settings(OPENAI_API_KEY="")
    def test_config_openai_test_endpoint_requires_api_key(self):
        response = self.client.post(
            reverse("aso:config_openai_test"),
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload["error_kind"], "missing_api_key")

    @override_settings(OPENAI_API_KEY="sk-env-123", OPENAI_MODEL="gpt-5-mini", OPENAI_AVAILABLE_MODELS=["gpt-5-mini"])
    @patch("openai.OpenAI")
    def test_config_openai_test_endpoint_surfaces_quota_errors(self, mock_openai):
        class QuotaError(Exception):
            pass

        err = QuotaError("insufficient_quota")
        err.status_code = 429
        err.request_id = "req_quota_123"
        err.body = {"error": {"code": "insufficient_quota"}}

        client = mock_openai.return_value
        client.with_options.return_value = client
        client.responses.create.side_effect = err

        response = self.client.post(
            reverse("aso:config_openai_test"),
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 429)
        payload = response.json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_kind"], "quota_or_rate_limit")
        self.assertEqual(payload["request_id"], "req_quota_123")


class DashboardServerSortTests(TestCase):
    def test_popularity_sort_is_global_before_pagination(self):
        app = App.objects.create(name="Sort App", track_id=424242)
        for i in range(1, 31):
            kw = Keyword.objects.create(keyword=f"sort-keyword-{i}", app=app)
            SearchResult.create_snapshot(
                keyword=kw,
                country="us",
                source=SearchResult.SOURCE_MANUAL_SEARCH,
                popularity_score=i,
                difficulty_score=50,
                difficulty_breakdown={},
                competitors_data=[],
                app_rank=None,
            )

        response = self.client.get(
            reverse("aso:dashboard"),
            {"sort": "popularity", "dir": "asc", "page": "2"},
        )
        self.assertEqual(response.status_code, 200)
        rows = list(response.context["history_results"])
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0].popularity_score, 26)
        self.assertEqual(rows[-1].popularity_score, 30)


class PromoRemovalTests(TestCase):
    def test_dashboard_has_no_top_banner_or_footer_cta(self):
        response = self.client.get(reverse("aso:dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Track your keywords over time")
        self.assertNotContains(response, "Try Free →")
        self.assertNotContains(response, "respectlytics.com/?utm_source=respectaso")


class AppStoreConnectEndpointTests(TestCase):
    def setUp(self):
        self.app = App.objects.create(name="ASC App", track_id=101010, asc_app_id="1234567890")
        cfg = RuntimeConfig.get_solo()
        cfg.asc_issuer_id = "issuer-id"
        cfg.asc_key_id = "key-id"
        cfg.asc_private_key_pem = "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----"
        cfg.asc_default_days_back = 30
        cfg.save()

    def test_status_endpoint_requires_app_id(self):
        response = self.client.get(reverse("aso:app_store_connect_status"))
        self.assertEqual(response.status_code, 400)

    def test_status_endpoint_returns_payload(self):
        ASCMetricDaily.objects.create(
            app=self.app,
            date=timezone.now().date(),
            country="us",
            impressions=100,
            product_page_views=80,
            app_units=20,
            conversion_rate=25.0,
        )
        response = self.client.get(reverse("aso:app_store_connect_status"), {"app_id": self.app.id})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["app_id"], self.app.id)
        self.assertEqual(payload["metric_rows"], 1)
        self.assertEqual(payload["asc_app_id"], "1234567890")

    @patch("aso.views.AppStoreConnectService.sync_app_metrics")
    def test_sync_endpoint_creates_run(self, mock_sync):
        mock_sync.return_value = 7
        response = self.client.post(
            reverse("aso:app_store_connect_sync"),
            data=json.dumps({"app_id": self.app.id, "days_back": 14}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["rows_upserted"], 7)
        run = ASCSyncRun.objects.first()
        self.assertIsNotNone(run)
        self.assertEqual(run.status, ASCSyncRun.STATUS_SUCCESS)

    def test_sync_endpoint_rejects_missing_asc_app_id(self):
        app = App.objects.create(name="No ASC", track_id=None, asc_app_id="")
        response = self.client.post(
            reverse("aso:app_store_connect_sync"),
            data=json.dumps({"app_id": app.id}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    @patch("aso.views.AppStoreConnectService.sync_app_metrics")
    def test_sync_endpoint_uses_track_id_fallback_for_asc_app_id(self, mock_sync):
        app = App.objects.create(name="Fallback ASC", track_id=202020, asc_app_id="")
        mock_sync.return_value = 3
        response = self.client.post(
            reverse("aso:app_store_connect_sync"),
            data=json.dumps({"app_id": app.id}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        mock_sync.assert_called_once()
        _args, kwargs = mock_sync.call_args
        self.assertEqual(kwargs["asc_app_id"], "202020")


class AICopilotEndpointTests(TestCase):
    def setUp(self):
        self.app = App.objects.create(name="Copilot App", track_id=303030, asc_app_id="999")
        self.run = AICopilotRun.objects.create(
            app=self.app,
            status=AICopilotRun.STATUS_SUCCESS,
            model="gpt-5-mini",
            country="us",
            input_snapshot_json={},
            feature_summary_json={},
        )
        self.recommendation = AICopilotRecommendation.objects.create(
            run=self.run,
            app=self.app,
            keyword="copilot keyword",
            action=AICopilotRecommendation.ACTION_ADD,
            rationale="good candidate",
            llm_confidence=0.7,
            score_market=0.6,
            score_rank_momentum=0.5,
            score_business_impact=0.4,
            score_coverage_gap=0.8,
            score_overall=0.62,
            evidence_json={"country": "us"},
        )
        self.variant = AICopilotMetadataVariant.objects.create(
            run=self.run,
            app=self.app,
            title="Copilot Title",
            subtitle="Subtitle",
            keyword_field="copilot,keyword",
            covered_keywords_json=["copilot keyword"],
            predicted_impact=0.66,
        )

    def test_list_endpoint_returns_copilot_payload(self):
        response = self.client.get(reverse("aso:ai_copilot_list"), {"app_id": self.app.id})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["app"]["id"], self.app.id)
        self.assertGreaterEqual(len(payload["recommendations"]), 1)
        self.assertGreaterEqual(len(payload["metadata_variants"]), 1)
        self.assertIn("progress_percent", payload["runs"][0])

    def test_status_endpoint_returns_latest_run(self):
        response = self.client.get(reverse("aso:ai_copilot_status"), {"app_id": self.app.id})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["app_id"], self.app.id)
        self.assertEqual(payload["run"]["id"], self.run.id)
        self.assertIn("events", payload["run"])
        self.assertIsInstance(payload["run"]["events"], list)

    @patch("aso.views._start_copilot_background_run")
    def test_generate_endpoint_returns_run_payload(self, mock_start):
        response = self.client.post(
            reverse("aso:ai_copilot_generate"),
            data=json.dumps({"app_id": self.app.id, "country": "us", "model": "gpt-5-mini"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertTrue(payload["has_running_run"])
        self.assertEqual(payload["run"]["status"], AICopilotRun.STATUS_RUNNING)
        mock_start.assert_called_once()

    @patch("aso.views._start_copilot_background_run")
    def test_generate_endpoint_passes_reasoning_effort_to_background(self, mock_start):
        response = self.client.post(
            reverse("aso:ai_copilot_generate"),
            data=json.dumps(
                {
                    "app_id": self.app.id,
                    "country": "us",
                    "model": "gpt-5-mini",
                    "reasoning_effort": "medium",
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 202)
        mock_start.assert_called_once()
        kwargs = mock_start.call_args.kwargs
        self.assertEqual(kwargs["effective_settings"]["reasoning_effort"], "medium")

    @patch("aso.views._start_copilot_background_run")
    def test_generate_endpoint_returns_conflict_when_run_is_already_running(self, mock_start):
        running = AICopilotRun.objects.create(
            app=self.app,
            status=AICopilotRun.STATUS_RUNNING,
            progress_percent=34,
            progress_stage="generating_recommendations",
            progress_detail="Running",
            model="gpt-5-mini",
            country="us",
            input_snapshot_json={},
            feature_summary_json={},
        )
        response = self.client.post(
            reverse("aso:ai_copilot_generate"),
            data=json.dumps({"app_id": self.app.id, "country": "us", "model": "gpt-5-mini"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 409)
        payload = response.json()
        self.assertEqual(payload["run"]["id"], running.id)
        self.assertTrue(payload["has_running_run"])
        mock_start.assert_not_called()

    @patch("aso.views._start_copilot_background_run")
    def test_generate_endpoint_marks_stale_running_run_failed_and_starts_new(self, mock_start):
        stale = AICopilotRun.objects.create(
            app=self.app,
            status=AICopilotRun.STATUS_RUNNING,
            progress_percent=34,
            progress_stage="generating_recommendations",
            progress_detail="Running",
            model="gpt-5-mini",
            country="us",
            input_snapshot_json={},
            feature_summary_json={},
        )
        AICopilotRun.objects.filter(pk=stale.id).update(started_at=timezone.now() - timedelta(hours=5))

        response = self.client.post(
            reverse("aso:ai_copilot_generate"),
            data=json.dumps({"app_id": self.app.id, "country": "us", "model": "gpt-5-mini"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 202)
        stale.refresh_from_db()
        self.assertEqual(stale.status, AICopilotRun.STATUS_FAILED)
        self.assertIn("stale threshold", stale.error.lower())
        mock_start.assert_called_once()

    def test_recommendation_accept_and_reject_endpoints(self):
        accept_resp = self.client.post(
            reverse("aso:ai_copilot_recommendation_accept", args=[self.recommendation.id])
        )
        self.assertEqual(accept_resp.status_code, 200)
        self.recommendation.refresh_from_db()
        self.assertEqual(self.recommendation.status, AICopilotRecommendation.STATUS_ACCEPTED)

        draft = AICopilotRecommendation.objects.create(
            run=self.run,
            app=self.app,
            keyword="reject me",
            action=AICopilotRecommendation.ACTION_WATCH,
            score_overall=0.2,
        )
        reject_resp = self.client.post(
            reverse("aso:ai_copilot_recommendation_reject", args=[draft.id])
        )
        self.assertEqual(reject_resp.status_code, 200)
        draft.refresh_from_db()
        self.assertEqual(draft.status, AICopilotRecommendation.STATUS_REJECTED)

    def test_metadata_accept_and_reject_endpoints(self):
        accept_resp = self.client.post(
            reverse("aso:ai_copilot_metadata_accept", args=[self.variant.id])
        )
        self.assertEqual(accept_resp.status_code, 200)
        self.variant.refresh_from_db()
        self.assertEqual(self.variant.status, AICopilotMetadataVariant.STATUS_ACCEPTED)

        draft = AICopilotMetadataVariant.objects.create(
            run=self.run,
            app=self.app,
            title="Reject Variant",
            subtitle="",
            keyword_field="one,two",
            predicted_impact=0.1,
        )
        reject_resp = self.client.post(
            reverse("aso:ai_copilot_metadata_reject", args=[draft.id])
        )
        self.assertEqual(reject_resp.status_code, 200)
        draft.refresh_from_db()
        self.assertEqual(draft.status, AICopilotMetadataVariant.STATUS_REJECTED)


class AICopilotServiceTimeoutRetryTests(TestCase):
    @patch("aso.copilot_ai_service._generate_with_openai")
    def test_generate_ai_copilot_retries_with_compact_snapshot_after_timeout(self, mock_generate):
        app = App.objects.create(name="Retry App", track_id=121212, asc_app_id="121212")
        for i in range(35):
            kw = Keyword.objects.create(keyword=f"retry keyword {i}", app=app)
            SearchResult.create_snapshot(
                keyword=kw,
                country="us",
                source=SearchResult.SOURCE_MANUAL_SEARCH,
                popularity_score=max(1, 62 - i),
                difficulty_score=min(99, 30 + i),
                difficulty_breakdown={},
                competitors_data=[],
                app_rank=12 + i,
            )
        output = CopilotOutput(
            recommendations=[
                RecommendationOut(
                    keyword="retry keyword plus",
                    action="add",
                    rationale="Timed-out primary request recovered with compact retry.",
                    llm_confidence=0.77,
                )
            ],
            metadata_variants=[
                MetadataVariantOut(
                    title="Retry Title",
                    subtitle="Retry subtitle",
                    keyword_field="retry,keyword",
                    covered_keywords=["retry keyword plus"],
                    predicted_impact=0.58,
                    rationale="Fallback output",
                )
            ],
        )
        mock_generate.side_effect = [
            AICopilotError(
                "Copilot generation failed: OpenAI request timed out. Retry, or increase OPENAI timeout in Config (OPENAI_TIMEOUT_SECONDS).",
                user_error=False,
            ),
            output,
        ]

        run = generate_ai_copilot(
            app,
            country="us",
            settings_obj=CopilotSettings(
                model="gpt-5-mini",
                api_key="sk-test-123",
                system_prompt="System prompt",
                user_prompt_template="{{SNAPSHOT_JSON}}",
            ),
        )

        self.assertEqual(run.status, AICopilotRun.STATUS_SUCCESS)
        self.assertEqual(mock_generate.call_count, 2)
        first_snapshot = mock_generate.call_args_list[0].kwargs["snapshot"]
        second_snapshot = mock_generate.call_args_list[1].kwargs["snapshot"]
        self.assertGreaterEqual(len(first_snapshot.get("feature_rows") or []), len(second_snapshot.get("feature_rows") or []))
        self.assertGreaterEqual(
            len(first_snapshot.get("existing_keywords") or []),
            len(second_snapshot.get("existing_keywords") or []),
        )


class AICopilotOpenAIRequestTests(TestCase):
    @patch("aso.copilot_ai_service._build_openai_client")
    def test_generate_with_openai_uses_low_reasoning_for_gpt5_models(self, mock_build_client):
        class DummyResponses:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(output_parsed=CopilotOutput())

        class DummyClient:
            def __init__(self):
                self.responses = DummyResponses()

            def with_options(self, **_kwargs):
                return self

        dummy = DummyClient()
        mock_build_client.return_value = dummy

        _generate_with_openai(
            snapshot={
                "feature_rows": [{"keyword": "one"}],
                "existing_keywords": ["one"],
            },
            settings_obj=CopilotSettings(
                model="gpt-5-mini",
                api_key="sk-test-123",
                system_prompt="System",
                user_prompt_template="{{SNAPSHOT_JSON}}",
            ),
            run_id=123,
            attempt="primary",
            timeout_seconds=20.0,
        )

        self.assertIsNotNone(dummy.responses.kwargs)
        self.assertEqual(dummy.responses.kwargs.get("reasoning", {}).get("effort"), "low")

    @patch("aso.copilot_ai_service._build_openai_client")
    def test_generate_with_openai_respects_configured_reasoning_effort(self, mock_build_client):
        class DummyResponses:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(output_parsed=CopilotOutput())

        class DummyClient:
            def __init__(self):
                self.responses = DummyResponses()

            def with_options(self, **_kwargs):
                return self

        dummy = DummyClient()
        mock_build_client.return_value = dummy

        _generate_with_openai(
            snapshot={
                "feature_rows": [{"keyword": "one"}],
                "existing_keywords": ["one"],
            },
            settings_obj=CopilotSettings(
                model="gpt-5-mini",
                api_key="sk-test-123",
                system_prompt="System",
                user_prompt_template="{{SNAPSHOT_JSON}}",
                reasoning_effort="high",
            ),
            run_id=123,
            attempt="primary",
            timeout_seconds=20.0,
        )

        self.assertIsNotNone(dummy.responses.kwargs)
        self.assertEqual(dummy.responses.kwargs.get("reasoning", {}).get("effort"), "high")

    @patch("aso.copilot_ai_service._build_openai_client")
    def test_generate_with_openai_parses_raw_output_text_without_chat_fallback(self, mock_build_client):
        raw_payload = {
            "recommendations": [
                {
                    "keyword": "raw keyword",
                    "action": "add",
                    "rationale": "raw parse",
                    "llm_confidence": 0.7,
                }
            ],
            "metadata_variants": [
                {
                    "title": "Raw Title",
                    "subtitle": "Raw subtitle",
                    "keyword_field": "raw,keyword",
                    "covered_keywords": ["raw keyword"],
                    "predicted_impact": 0.6,
                    "rationale": "raw variant",
                }
            ],
        }

        class DummyResponses:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    output_parsed=None,
                    output_text=json.dumps(raw_payload),
                    output=[],
                    status="completed",
                    _request_id="req_test_raw",
                )

        class DummyChatCompletions:
            def parse(self, **_kwargs):
                raise AssertionError("chat fallback should not be called when raw responses output is valid")

        class DummyClient:
            def __init__(self):
                self.responses = DummyResponses()
                self.chat = SimpleNamespace(completions=DummyChatCompletions())

            def with_options(self, **_kwargs):
                return self

        dummy = DummyClient()
        mock_build_client.return_value = dummy

        output = _generate_with_openai(
            snapshot={
                "feature_rows": [{"keyword": "one"}],
                "existing_keywords": ["one"],
            },
            settings_obj=CopilotSettings(
                model="gpt-5-mini",
                api_key="sk-test-123",
                system_prompt="System",
                user_prompt_template="{{SNAPSHOT_JSON}}",
                reasoning_effort="medium",
            ),
            run_id=123,
            attempt="primary",
            timeout_seconds=20.0,
        )

        self.assertEqual(output.recommendations[0].keyword, "raw keyword")
