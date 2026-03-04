# Generated manually for runtime in-app configuration.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("aso", "0007_searchresult_source_and_ai_models"),
    ]

    operations = [
        migrations.CreateModel(
            name="RuntimeConfig",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("singleton_key", models.PositiveSmallIntegerField(default=1, editable=False, unique=True)),
                ("openai_api_key", models.CharField(blank=True, default="", max_length=255)),
                ("openai_default_model", models.CharField(blank=True, default="", max_length=100)),
                (
                    "openai_available_models",
                    models.TextField(
                        blank=True,
                        default="",
                        help_text="Comma-separated list of models shown in the AI Suggestions dropdown.",
                    ),
                ),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "verbose_name": "Runtime Config",
                "verbose_name_plural": "Runtime Config",
            },
        ),
    ]
