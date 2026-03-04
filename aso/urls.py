from django.urls import path

from . import views

app_name = "aso"

urlpatterns = [
    path("", views.dashboard_view, name="dashboard"),
    path("ai/suggestions/", views.ai_suggestions_view, name="ai_suggestions"),
    path("ai/suggestions/list/", views.ai_suggestions_list_view, name="ai_suggestions_list"),
    path("ai/suggestions/generate/", views.ai_suggestions_generate_view, name="ai_suggestions_generate"),
    path("ai/suggestions/<int:suggestion_id>/accept/", views.ai_suggestion_accept_view, name="ai_suggestion_accept"),
    path("ai/suggestions/<int:suggestion_id>/reject/", views.ai_suggestion_reject_view, name="ai_suggestion_reject"),
    path("ai/copilot/list/", views.ai_copilot_list_view, name="ai_copilot_list"),
    path("ai/copilot/generate/", views.ai_copilot_generate_view, name="ai_copilot_generate"),
    path(
        "ai/copilot/recommendations/<int:recommendation_id>/accept/",
        views.ai_copilot_recommendation_accept_view,
        name="ai_copilot_recommendation_accept",
    ),
    path(
        "ai/copilot/recommendations/<int:recommendation_id>/reject/",
        views.ai_copilot_recommendation_reject_view,
        name="ai_copilot_recommendation_reject",
    ),
    path(
        "ai/copilot/metadata/<int:variant_id>/accept/",
        views.ai_copilot_metadata_accept_view,
        name="ai_copilot_metadata_accept",
    ),
    path(
        "ai/copilot/metadata/<int:variant_id>/reject/",
        views.ai_copilot_metadata_reject_view,
        name="ai_copilot_metadata_reject",
    ),
    path("integrations/app-store-connect/sync/", views.app_store_connect_sync_view, name="app_store_connect_sync"),
    path(
        "integrations/app-store-connect/status/",
        views.app_store_connect_status_view,
        name="app_store_connect_status",
    ),
    path("config/", views.config_view, name="config"),
    path("methodology/", views.methodology_view, name="methodology"),
    path("setup/", views.setup_view, name="setup"),
    path("search/", views.search_view, name="search"),
    path("opportunity/", views.opportunity_view, name="opportunity"),
    path("opportunity/search/", views.opportunity_search_view, name="opportunity_search"),
    path("opportunity/save/", views.opportunity_save_view, name="opportunity_save"),
    path("export/history.csv", views.export_history_csv_view, name="export_history_csv"),
    path("apps/", views.apps_view, name="apps"),
    path("apps/lookup/", views.app_lookup_view, name="app_lookup"),
    path("apps/<int:app_id>/delete/", views.app_delete_view, name="app_delete"),
    path("keywords/<int:keyword_id>/delete/", views.keyword_delete_view, name="keyword_delete"),
    path("results/<int:result_id>/delete/", views.result_delete_view, name="result_delete"),
    path("keywords/bulk-delete/", views.keywords_bulk_delete_view, name="keywords_bulk_delete"),
    path("keywords/<int:keyword_id>/refresh/", views.keyword_refresh_view, name="keyword_refresh"),
    path("keywords/bulk-refresh/", views.keywords_bulk_refresh_view, name="keywords_bulk_refresh"),
    path("auto-refresh/status/", views.auto_refresh_status_view, name="auto_refresh_status"),
    path("keywords/<int:keyword_id>/trend/", views.keyword_trend_view, name="keyword_trend"),
    path("version-check/", views.version_check_view, name="version_check"),
]
