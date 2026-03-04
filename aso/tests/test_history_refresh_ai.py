from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from aso.ai_service import AISuggestionError, accept_ai_suggestion, generate_ai_suggestions
from aso.models import AISuggestion, AISuggestionRun, App, Keyword, RefreshRun, SearchResult
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
            data='{"app_id": %d}' % self.app.id,
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["run"]["id"], self.run.id)

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
        self.assertContains(response, "AI Keyword Suggestions")

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
