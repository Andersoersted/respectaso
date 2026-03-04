from django import forms

from .models import App


class AppForm(forms.ModelForm):
    """Form for creating/editing an App."""

    class Meta:
        model = App
        fields = ["name", "bundle_id"]
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
    openai_default_model = forms.CharField(
        required=False,
        widget=forms.TextInput(
            attrs={
                "class": "w-full bg-slate-700 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500",
                "placeholder": "gpt-4.1-mini",
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
        help_text="Comma-separated models shown in the AI Suggestions model dropdown.",
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

    def __init__(self, *args, config, **kwargs):
        self.config = config
        super().__init__(*args, **kwargs)
        self.fields["openai_default_model"].initial = config.openai_default_model
        self.fields["openai_available_models"].initial = config.openai_available_models
        self.fields["ai_enable_online_context"].initial = bool(config.ai_enable_online_context)
        self.fields["ai_online_top_apps_per_country"].initial = config.ai_online_top_apps_per_country
        self.fields["ai_history_rows_max"].initial = config.ai_history_rows_max
        self.fields["ai_system_prompt"].initial = config.ai_system_prompt
        self.fields["ai_user_prompt_template"].initial = config.ai_user_prompt_template

    def save(self):
        if self.cleaned_data.get("clear_openai_api_key"):
            self.config.openai_api_key = ""
        else:
            incoming_key = (self.cleaned_data.get("openai_api_key") or "").strip()
            if incoming_key:
                self.config.openai_api_key = incoming_key

        self.config.openai_default_model = (
            self.cleaned_data.get("openai_default_model") or ""
        ).strip()
        self.config.openai_available_models = (
            self.cleaned_data.get("openai_available_models") or ""
        ).strip()
        self.config.ai_enable_online_context = self.cleaned_data.get("ai_enable_online_context")
        self.config.ai_online_top_apps_per_country = self.cleaned_data.get("ai_online_top_apps_per_country")
        self.config.ai_history_rows_max = self.cleaned_data.get("ai_history_rows_max")
        self.config.ai_system_prompt = (self.cleaned_data.get("ai_system_prompt") or "").strip()
        self.config.ai_user_prompt_template = (
            self.cleaned_data.get("ai_user_prompt_template") or ""
        ).strip()
        self.config.save()
        return self.config
