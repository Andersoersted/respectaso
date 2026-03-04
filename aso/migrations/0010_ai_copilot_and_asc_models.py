# Generated manually for AI Copilot + App Store Connect integration models.

import decimal

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("aso", "0009_runtimeconfig_prompt_and_online_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="app",
            name="asc_app_id",
            field=models.CharField(
                blank=True,
                db_index=True,
                default="",
                help_text="App Store Connect app ID used for analytics sync.",
                max_length=64,
            ),
        ),
        migrations.AlterField(
            model_name="runtimeconfig",
            name="openai_available_models",
            field=models.TextField(
                blank=True,
                default="",
                help_text="Comma-separated list of models shown in the AI Copilot dropdown.",
            ),
        ),
        migrations.AlterField(
            model_name="runtimeconfig",
            name="ai_enable_online_context",
            field=models.BooleanField(
                blank=True,
                default=None,
                help_text="Enable online market enrichment for AI Copilot.",
                null=True,
            ),
        ),
        migrations.AddField(
            model_name="runtimeconfig",
            name="asc_default_days_back",
            field=models.PositiveSmallIntegerField(
                blank=True,
                help_text="Default number of days to sync from App Store Connect.",
                null=True,
            ),
        ),
        migrations.AddField(
            model_name="runtimeconfig",
            name="asc_issuer_id",
            field=models.CharField(
                blank=True,
                default="",
                help_text="App Store Connect API issuer ID.",
                max_length=128,
            ),
        ),
        migrations.AddField(
            model_name="runtimeconfig",
            name="asc_key_id",
            field=models.CharField(
                blank=True,
                default="",
                help_text="App Store Connect API key ID (kid).",
                max_length=32,
            ),
        ),
        migrations.AddField(
            model_name="runtimeconfig",
            name="asc_private_key_pem",
            field=models.TextField(
                blank=True,
                default="",
                help_text="App Store Connect private key PEM content.",
            ),
        ),
        migrations.AlterField(
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
                    ("ai_copilot", "AI Copilot"),
                ],
                db_index=True,
                default="manual_search",
                max_length=32,
            ),
        ),
        migrations.CreateModel(
            name="AICopilotRun",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                (
                    "status",
                    models.CharField(
                        choices=[("running", "Running"), ("success", "Success"), ("failed", "Failed")],
                        default="running",
                        max_length=16,
                    ),
                ),
                ("model", models.CharField(max_length=100)),
                ("country", models.CharField(default="us", max_length=5)),
                ("input_snapshot_json", models.JSONField(default=dict)),
                ("feature_summary_json", models.JSONField(default=dict)),
                ("error", models.TextField(blank=True, default="")),
                ("started_at", models.DateTimeField(auto_now_add=True)),
                ("finished_at", models.DateTimeField(blank=True, null=True)),
                (
                    "app",
                    models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="ai_copilot_runs", to="aso.app"),
                ),
            ],
            options={"ordering": ["-started_at"]},
        ),
        migrations.CreateModel(
            name="ASCSyncRun",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                (
                    "status",
                    models.CharField(
                        choices=[("running", "Running"), ("success", "Success"), ("failed", "Failed")],
                        default="running",
                        max_length=16,
                    ),
                ),
                ("days_back", models.PositiveSmallIntegerField(default=30)),
                ("rows_upserted", models.PositiveIntegerField(default=0)),
                ("started_at", models.DateTimeField(auto_now_add=True)),
                ("finished_at", models.DateTimeField(blank=True, null=True)),
                ("error", models.TextField(blank=True, default="")),
                ("app", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="asc_sync_runs", to="aso.app")),
            ],
            options={"ordering": ["-started_at"]},
        ),
        migrations.CreateModel(
            name="ASCMetricDaily",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("date", models.DateField(db_index=True)),
                ("country", models.CharField(default="us", max_length=5)),
                ("impressions", models.IntegerField(blank=True, null=True)),
                ("product_page_views", models.IntegerField(blank=True, null=True)),
                ("app_units", models.IntegerField(blank=True, null=True)),
                ("conversion_rate", models.FloatField(blank=True, null=True)),
                ("proceeds", models.DecimalField(decimal_places=2, default=decimal.Decimal("0.00"), max_digits=14)),
                ("raw_json", models.JSONField(default=dict)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("app", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="asc_metrics", to="aso.app")),
            ],
            options={"ordering": ["-date", "country"]},
        ),
        migrations.AddConstraint(
            model_name="ascmetricdaily",
            constraint=models.UniqueConstraint(
                fields=("app", "date", "country"),
                name="aso_ascmetric_app_date_country_uniq",
            ),
        ),
        migrations.AddIndex(
            model_name="ascmetricdaily",
            index=models.Index(fields=["app", "country", "-date"], name="aso_ascmetric_lookup_idx"),
        ),
        migrations.CreateModel(
            name="AICopilotRecommendation",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                (
                    "action",
                    models.CharField(
                        choices=[
                            ("add", "Add"),
                            ("promote", "Promote"),
                            ("watch", "Watch"),
                            ("deprioritize", "Deprioritize"),
                        ],
                        default="watch",
                        max_length=16,
                    ),
                ),
                ("keyword", models.CharField(max_length=200)),
                ("rationale", models.TextField(blank=True, default="")),
                ("llm_confidence", models.FloatField(default=0.0)),
                ("score_market", models.FloatField(default=0.0)),
                ("score_rank_momentum", models.FloatField(default=0.0)),
                ("score_business_impact", models.FloatField(default=0.0)),
                ("score_coverage_gap", models.FloatField(default=0.0)),
                ("score_overall", models.FloatField(db_index=True, default=0.0)),
                ("evidence_json", models.JSONField(default=dict)),
                (
                    "status",
                    models.CharField(
                        choices=[("draft", "Draft"), ("accepted", "Accepted"), ("rejected", "Rejected")],
                        db_index=True,
                        default="draft",
                        max_length=16,
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("resolved_at", models.DateTimeField(blank=True, null=True)),
                (
                    "app",
                    models.ForeignKey(
                        on_delete=models.deletion.CASCADE,
                        related_name="ai_copilot_recommendations",
                        to="aso.app",
                    ),
                ),
                (
                    "created_keyword",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=models.deletion.SET_NULL,
                        related_name="ai_copilot_recommendations",
                        to="aso.keyword",
                    ),
                ),
                (
                    "run",
                    models.ForeignKey(
                        on_delete=models.deletion.CASCADE,
                        related_name="recommendations",
                        to="aso.aicopilotrun",
                    ),
                ),
            ],
            options={"ordering": ["-score_overall", "-created_at"]},
        ),
        migrations.AddConstraint(
            model_name="aicopilotrecommendation",
            constraint=models.UniqueConstraint(fields=("run", "keyword"), name="aso_copilot_reco_run_keyword_uniq"),
        ),
        migrations.CreateModel(
            name="AICopilotMetadataVariant",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("title", models.CharField(max_length=60)),
                ("subtitle", models.CharField(blank=True, default="", max_length=60)),
                ("keyword_field", models.CharField(blank=True, default="", max_length=180)),
                ("covered_keywords_json", models.JSONField(default=list)),
                ("predicted_impact", models.FloatField(default=0.0)),
                ("rationale", models.TextField(blank=True, default="")),
                (
                    "status",
                    models.CharField(
                        choices=[("draft", "Draft"), ("accepted", "Accepted"), ("rejected", "Rejected")],
                        db_index=True,
                        default="draft",
                        max_length=16,
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("resolved_at", models.DateTimeField(blank=True, null=True)),
                (
                    "app",
                    models.ForeignKey(
                        on_delete=models.deletion.CASCADE,
                        related_name="ai_copilot_metadata_variants",
                        to="aso.app",
                    ),
                ),
                (
                    "run",
                    models.ForeignKey(
                        on_delete=models.deletion.CASCADE,
                        related_name="metadata_variants",
                        to="aso.aicopilotrun",
                    ),
                ),
            ],
            options={"ordering": ["-predicted_impact", "-created_at"]},
        ),
    ]
