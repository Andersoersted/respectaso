from django.conf import settings
from django import forms

from .models import App


class AppForm(forms.ModelForm):
    """Form for creating/editing an App."""

    class Meta:
        model = App
        fields = ["name", "bundle_id", "asc_app_id"]
        widgets = {
            "name": forms.TextInput(
                attrs={
                    "class": "w-full bg-slate-700 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500",
                    "placeholder": "My iOS App",
                }
            ),
            "bundle_id": forms.TextInput(
                attrs={
                    "class": "w-full bg-slate-700 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500",
                    "placeholder": "com.example.myapp (optional)",
                }
            ),
            "asc_app_id": forms.TextInput(
                attrs={
                    "class": "w-full bg-slate-700 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500",
                    "placeholder": "1234567890 (App Store Connect app ID, optional)",
                }
            ),
        }


COUNTRY_CHOICES = [
    ("us", "🇺🇸 United States"),
    ("gb", "🇬🇧 United Kingdom"),
    ("ca", "🇨🇦 Canada"),
    ("au", "🇦🇺 Australia"),
    ("de", "🇩🇪 Germany"),
    ("fr", "🇫🇷 France"),
    ("jp", "🇯🇵 Japan"),
    ("kr", "🇰🇷 South Korea"),
    ("cn", "🇨🇳 China"),
    ("br", "🇧🇷 Brazil"),
    ("in", "🇮🇳 India"),
    ("mx", "🇲🇽 Mexico"),
    ("es", "🇪🇸 Spain"),
    ("it", "🇮🇹 Italy"),
    ("nl", "🇳🇱 Netherlands"),
    ("se", "🇸🇪 Sweden"),
    ("no", "🇳🇴 Norway"),
    ("dk", "🇩🇰 Denmark"),
    ("fi", "🇫🇮 Finland"),
    ("pt", "🇵🇹 Portugal"),
    ("ru", "🇷🇺 Russia"),
    ("tr", "🇹🇷 Turkey"),
    ("sa", "🇸🇦 Saudi Arabia"),
    ("ae", "🇦🇪 UAE"),
    ("sg", "🇸🇬 Singapore"),
    ("th", "🇹🇭 Thailand"),
    ("id", "🇮🇩 Indonesia"),
    ("ph", "🇵🇭 Philippines"),
    ("vn", "🇻🇳 Vietnam"),
    ("tw", "🇹🇼 Taiwan"),
]

OPENAI_REASONING_CHOICES = [
    ("", "Auto"),
    ("none", "None"),
    ("minimal", "Minimal"),
    ("low", "Low"),
    ("medium", "Medium"),
    ("high", "High"),
    ("xhigh", "XHigh"),
]


class KeywordSearchForm(forms.Form):
    """Form for searching keywords."""

    keywords = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "class": "w-full bg-slate-700 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500",
                "placeholder": "meditation app, fitness tracker, sleep sounds",
                "autofocus": True,
            }
        ),
        label="Keywords",
        help_text="Enter one or more keywords, separated by commas (max 20).",
    )
    app_id = forms.IntegerField(
        required=False,
        widget=forms.HiddenInput(),
    )
    countries = forms.CharField(
        required=False,
        widget=forms.HiddenInput(),
        help_text="Comma-separated country codes (max 5).",
    )

    def clean_countries(self):
        """Parse and validate comma-separated country codes."""
        raw = self.cleaned_data.get("countries", "").strip()
        if not raw:
            return ["us"]
        valid_codes = {code for code, _ in COUNTRY_CHOICES}
        codes = [c.strip().lower() for c in raw.split(",") if c.strip()]
        codes = [c for c in codes if c in valid_codes]
        if not codes:
            return ["us"]
        return codes[:5]  # Max 5 countries


class OpportunitySearchForm(forms.Form):
    """Form for the Country Opportunity Finder — single keyword, all countries."""

    keyword = forms.CharField(
        max_length=200,
        widget=forms.TextInput(
            attrs={
                "class": "w-full bg-slate-700 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500",
                "placeholder": "fitness tracker",
                "autofocus": True,
            }
        ),
    )
    app_id = forms.IntegerField(
        required=False,
        widget=forms.HiddenInput(),
    )


class RuntimeConfigForm(forms.Form):
    openai_api_key = forms.CharField(
        required=False,
        widget=forms.PasswordInput(
            render_value=False,
            attrs={
                "class": "w-full bg-slate-700 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500",
                "placeholder": "sk-... (leave blank to keep current key)",
            },
        ),
        label="OpenAI API Key",
    )
    clear_openai_api_key = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(
            attrs={
                "class": "rounded border-white/20 bg-slate-700 text-purple-500 focus:ring-purple-500",
            }
        ),
        label="Clear stored API key override",
    )
    openai_default_model = forms.ChoiceField(
        required=False,
        choices=[],
        widget=forms.Select(
            attrs={
                "class": "w-full bg-slate-700 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500",
            }
        ),
        label="Default AI Model",
    )
    openai_available_models = forms.CharField(
        required=False,
        widget=forms.TextInput(
            attrs={
                "class": "w-full bg-slate-700 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500",
                "placeholder": "gpt-4.1-mini,gpt-4.1,gpt-4o-mini",
            }
        ),
        label="Available Models",
        help_text="Comma-separated models shown in the AI Copilot model dropdown.",
    )
    openai_reasoning_effort = forms.ChoiceField(
        required=False,
        choices=OPENAI_REASONING_CHOICES,
        widget=forms.Select(
            attrs={
                "class": "w-full bg-slate-700 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500",
            }
        ),
        label="Default Reasoning Effort",
        help_text="Blank uses model defaults. Lower effort is usually faster.",
    )
    ai_enable_online_context = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(
            attrs={
                "class": "rounded border-white/20 bg-slate-700 text-purple-500 focus:ring-purple-500",
            }
        ),
        label="Enable online market context",
    )
    ai_online_top_apps_per_country = forms.IntegerField(
        required=False,
        min_value=5,
        max_value=50,
        widget=forms.NumberInput(
            attrs={
                "class": "w-full bg-slate-700 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500",
                "placeholder": "20",
            }
        ),
        label="Online top apps per country (5-50)",
    )
    ai_history_rows_max = forms.IntegerField(
        required=False,
        min_value=50,
        max_value=5000,
        widget=forms.NumberInput(
            attrs={
                "class": "w-full bg-slate-700 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500",
                "placeholder": "300",
            }
        ),
        label="History rows sent to AI (50-5000)",
    )
    ai_system_prompt = forms.CharField(
        required=False,
        widget=forms.Textarea(
            attrs={
                "class": "w-full min-h-[120px] bg-slate-700 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500",
                "placeholder": "System prompt for AI behavior",
            }
        ),
        label="System Prompt",
    )
    ai_user_prompt_template = forms.CharField(
        required=False,
        widget=forms.Textarea(
            attrs={
                "class": "w-full min-h-[220px] bg-slate-700 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500",
                "placeholder": "Use {{SNAPSHOT_JSON}} and {{ONLINE_CONTEXT_JSON}} placeholders",
            }
        ),
        label="User Prompt Template",
        help_text="Use placeholders {{SNAPSHOT_JSON}} and {{ONLINE_CONTEXT_JSON}}.",
    )
    asc_issuer_id = forms.CharField(
        required=False,
        widget=forms.TextInput(
            attrs={
                "class": "w-full bg-slate-700 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500",
                "placeholder": "Issuer ID",
            }
        ),
        label="App Store Connect Issuer ID",
    )
    asc_key_id = forms.CharField(
        required=False,
        widget=forms.TextInput(
            attrs={
                "class": "w-full bg-slate-700 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500",
                "placeholder": "Key ID (kid)",
            }
        ),
        label="App Store Connect Key ID",
    )
    asc_private_key_pem = forms.CharField(
        required=False,
        widget=forms.Textarea(
            attrs={
                "class": "w-full min-h-[140px] bg-slate-700 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500",
                "placeholder": "-----BEGIN PRIVATE KEY----- ...",
            }
        ),
        label="App Store Connect Private Key (PEM)",
    )
    clear_asc_private_key = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(
            attrs={
                "class": "rounded border-white/20 bg-slate-700 text-purple-500 focus:ring-purple-500",
            }
        ),
        label="Clear stored App Store Connect private key",
    )
    asc_default_days_back = forms.IntegerField(
        required=False,
        min_value=1,
        max_value=365,
        widget=forms.NumberInput(
            attrs={
                "class": "w-full bg-slate-700 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500",
                "placeholder": "30",
            }
        ),
        label="ASC sync default days back (1-365)",
    )

    def __init__(self, *args, config, **kwargs):
        self.config = config
        super().__init__(*args, **kwargs)
        available_model_csv = config.openai_available_models.strip() or ",".join(
            settings.OPENAI_AVAILABLE_MODELS
        )
        models = []
        for raw in available_model_csv.split(","):
            value = raw.strip()
            if value and value not in models:
                models.append(value)
        default_model = config.openai_default_model.strip() or settings.OPENAI_MODEL
        if default_model and default_model not in models:
            models.insert(0, default_model)
        self.fields["openai_default_model"].choices = [("", "(Use environment default)")] + [
            (m, m) for m in models
        ]
        self.fields["openai_default_model"].initial = default_model
        self.fields["openai_available_models"].initial = available_model_csv
        self.fields["openai_reasoning_effort"].initial = (config.openai_reasoning_effort or "").strip().lower()
        if config.ai_enable_online_context is None:
            self.fields["ai_enable_online_context"].initial = settings.AI_ENABLE_ONLINE_CONTEXT
        else:
            self.fields["ai_enable_online_context"].initial = bool(config.ai_enable_online_context)
        self.fields["ai_online_top_apps_per_country"].initial = (
            config.ai_online_top_apps_per_country or settings.AI_ONLINE_TOP_APPS_PER_COUNTRY
        )
        self.fields["ai_history_rows_max"].initial = (
            config.ai_history_rows_max or settings.AI_HISTORY_ROWS_MAX
        )
        self.fields["ai_system_prompt"].initial = config.ai_system_prompt or settings.AI_SYSTEM_PROMPT
        self.fields["ai_user_prompt_template"].initial = (
            config.ai_user_prompt_template or settings.AI_USER_PROMPT_TEMPLATE
        )
        self.fields["asc_issuer_id"].initial = config.asc_issuer_id
        self.fields["asc_key_id"].initial = config.asc_key_id
        self.fields["asc_default_days_back"].initial = config.asc_default_days_back or settings.ASC_DEFAULT_DAYS_BACK

    def save(self):
        incoming_key = (self.cleaned_data.get("openai_api_key") or "").strip()
        if incoming_key:
            # Typed key should always win over a stale/checked clear box.
            self.config.openai_api_key = incoming_key
        elif self.cleaned_data.get("clear_openai_api_key"):
            self.config.openai_api_key = ""

        self.config.openai_default_model = (
            self.cleaned_data.get("openai_default_model") or ""
        ).strip()
        self.config.openai_available_models = (
            self.cleaned_data.get("openai_available_models") or ""
        ).strip()
        self.config.openai_reasoning_effort = (
            self.cleaned_data.get("openai_reasoning_effort") or ""
        ).strip().lower()
        self.config.ai_enable_online_context = self.cleaned_data.get("ai_enable_online_context")
        self.config.ai_online_top_apps_per_country = self.cleaned_data.get("ai_online_top_apps_per_country")
        self.config.ai_history_rows_max = self.cleaned_data.get("ai_history_rows_max")
        self.config.ai_system_prompt = (self.cleaned_data.get("ai_system_prompt") or "").strip()
        self.config.ai_user_prompt_template = (
            self.cleaned_data.get("ai_user_prompt_template") or ""
        ).strip()
        self.config.asc_issuer_id = (self.cleaned_data.get("asc_issuer_id") or "").strip()
        self.config.asc_key_id = (self.cleaned_data.get("asc_key_id") or "").strip()
        incoming_private_key = (self.cleaned_data.get("asc_private_key_pem") or "").strip()
        if incoming_private_key:
            # Same precedence rule as OpenAI key above.
            self.config.asc_private_key_pem = incoming_private_key
        elif self.cleaned_data.get("clear_asc_private_key"):
            self.config.asc_private_key_pem = ""
        self.config.asc_default_days_back = self.cleaned_data.get("asc_default_days_back")
        self.config.save()
        return self.config
