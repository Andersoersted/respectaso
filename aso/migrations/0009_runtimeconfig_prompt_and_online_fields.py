# Generated manually for runtime prompt tuning and online enrichment settings.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("aso", "0008_runtimeconfig"),
    ]

    operations = [
        migrations.AddField(
            model_name="runtimeconfig",
            name="ai_enable_online_context",
            field=models.BooleanField(
                blank=True,
                default=None,
                help_text="Enable online market enrichment for AI suggestions.",
                null=True,
            ),
        ),
        migrations.AddField(
            model_name="runtimeconfig",
            name="ai_history_rows_max",
            field=models.PositiveIntegerField(
                blank=True,
                help_text="Maximum number of historical snapshots passed to AI context.",
                null=True,
            ),
        ),
        migrations.AddField(
            model_name="runtimeconfig",
            name="ai_online_top_apps_per_country",
            field=models.PositiveSmallIntegerField(
                blank=True,
                help_text="Top free apps fetched per country when online context is enabled (5-50).",
                null=True,
            ),
        ),
        migrations.AddField(
            model_name="runtimeconfig",
            name="ai_system_prompt",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AddField(
            model_name="runtimeconfig",
            name="ai_user_prompt_template",
            field=models.TextField(blank=True, default=""),
        ),
    ]
