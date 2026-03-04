# Generated manually for append-only history, refresh runs, and AI suggestions.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("aso", "0006_alter_searchresult_popularity_score"),
    ]

    operations = [
        migrations.AddField(
            model_name="searchresult",
            name="source",
            field=models.CharField(
                choices=[
                    ("manual_search", "Manual Search"),
                    ("manual_refresh", "Manual Refresh"),
                    ("bulk_refresh", "Bulk Refresh"),
                    ("daily_refresh", "Daily Refresh"),
                    ("opportunity_save", "Opportunity Save"),
                    ("ai_suggestion", "AI Suggestion"),
                ],
                db_index=True,
                default="manual_search",
                max_length=32,
            ),
        ),
        migrations.AddIndex(
            model_name="searchresult",
            index=models.Index(
                fields=["keyword", "country", "-searched_at"],
                name="aso_kw_country_srch_idx",
            ),
        ),
        migrations.CreateModel(
            name="RefreshRun",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                (
                    "trigger",
                    models.CharField(
                        choices=[
                            ("cron", "Cron"),
                            ("manual", "Manual"),
                            ("thread", "Thread Scheduler"),
                        ],
                        max_length=16,
                    ),
                ),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("running", "Running"),
                            ("success", "Success"),
                            ("failed", "Failed"),
                        ],
                        default="running",
                        max_length=16,
                    ),
                ),
                ("total", models.PositiveIntegerField(default=0)),
                ("completed", models.PositiveIntegerField(default=0)),
                ("current_keyword", models.CharField(blank=True, default="", max_length=255)),
                ("started_at", models.DateTimeField(auto_now_add=True)),
                ("finished_at", models.DateTimeField(blank=True, null=True)),
                ("error", models.TextField(blank=True, default="")),
            ],
            options={
                "ordering": ["-started_at"],
            },
        ),
        migrations.CreateModel(
            name="AISuggestionRun",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("running", "Running"),
                            ("success", "Success"),
                            ("failed", "Failed"),
                        ],
                        default="running",
                        max_length=16,
                    ),
                ),
                ("model", models.CharField(max_length=100)),
                ("countries_json", models.JSONField(default=list)),
                ("input_snapshot_json", models.JSONField(default=dict)),
                ("error", models.TextField(blank=True, default="")),
                ("started_at", models.DateTimeField(auto_now_add=True)),
                ("finished_at", models.DateTimeField(blank=True, null=True)),
                (
                    "app",
                    models.ForeignKey(on_delete=models.CASCADE, related_name="ai_suggestion_runs", to="aso.app"),
                ),
            ],
            options={
                "ordering": ["-started_at"],
            },
        ),
        migrations.CreateModel(
            name="AISuggestion",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("keyword", models.CharField(max_length=200)),
                ("intent", models.CharField(blank=True, default="", max_length=200)),
                ("rationale", models.TextField(blank=True, default="")),
                ("confidence", models.FloatField(default=0.0)),
                ("score_market", models.FloatField(default=0.0)),
                ("score_history", models.FloatField(default=0.5)),
                ("score_overall", models.FloatField(db_index=True, default=0.0)),
                ("market_metrics_json", models.JSONField(default=dict)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("draft", "Draft"),
                            ("accepted", "Accepted"),
                            ("rejected", "Rejected"),
                        ],
                        db_index=True,
                        default="draft",
                        max_length=16,
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("resolved_at", models.DateTimeField(blank=True, null=True)),
                (
                    "app",
                    models.ForeignKey(on_delete=models.CASCADE, related_name="ai_suggestions", to="aso.app"),
                ),
                (
                    "created_keyword",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=models.SET_NULL,
                        related_name="ai_suggestions",
                        to="aso.keyword",
                    ),
                ),
                (
                    "run",
                    models.ForeignKey(on_delete=models.CASCADE, related_name="suggestions", to="aso.aisuggestionrun"),
                ),
            ],
            options={
                "ordering": ["-score_overall", "-created_at"],
                "unique_together": {("run", "keyword")},
            },
        ),
    ]
