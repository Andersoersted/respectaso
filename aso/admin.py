from django.contrib import admin

from .models import AISuggestion, AISuggestionRun, App, Keyword, RefreshRun, SearchResult


@admin.register(App)
class AppAdmin(admin.ModelAdmin):
    list_display = ("name", "bundle_id", "created_at")
    search_fields = ("name", "bundle_id")


@admin.register(Keyword)
class KeywordAdmin(admin.ModelAdmin):
    list_display = ("keyword", "app", "created_at")
    list_filter = ("app",)
    search_fields = ("keyword",)


@admin.register(SearchResult)
class SearchResultAdmin(admin.ModelAdmin):
    list_display = (
        "keyword",
        "popularity_score",
        "difficulty_score",
        "source",
        "country",
        "searched_at",
    )
    list_filter = ("source", "country", "searched_at")
    readonly_fields = ("searched_at",)


@admin.register(RefreshRun)
class RefreshRunAdmin(admin.ModelAdmin):
    list_display = ("id", "trigger", "status", "total", "completed", "started_at", "finished_at")
    list_filter = ("trigger", "status", "started_at")
    search_fields = ("current_keyword", "error")
    readonly_fields = ("started_at", "finished_at")


@admin.register(AISuggestionRun)
class AISuggestionRunAdmin(admin.ModelAdmin):
    list_display = ("id", "app", "status", "model", "started_at", "finished_at")
    list_filter = ("status", "model", "started_at")
    search_fields = ("app__name", "error")
    readonly_fields = ("started_at", "finished_at")


@admin.register(AISuggestion)
class AISuggestionAdmin(admin.ModelAdmin):
    list_display = ("keyword", "app", "status", "score_overall", "confidence", "created_at")
    list_filter = ("status", "created_at")
    search_fields = ("keyword", "app__name")
    readonly_fields = ("created_at", "resolved_at")
