import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Current version — update on each release
VERSION = "1.4.0"

# Load .env from persistent data volume (auto-generated SECRET_KEY lives there)
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

env_file = DATA_DIR / ".env"
if env_file.exists():
    load_dotenv(env_file)

SECRET_KEY = os.environ.get(
    "SECRET_KEY",
    "django-insecure-dev-key-change-me-in-production",
)

DEBUG = os.environ.get("DEBUG", "True").lower() in ("true", "1", "yes")

DEFAULT_ALLOWED_HOSTS = [
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "respectaso.private",
]


def csv_env(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name, "")
    if not value.strip():
        return default
    return [part.strip() for part in value.split(",") if part.strip()]


def int_env(name: str, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    raw = os.environ.get(name)
    try:
        value = int(raw) if raw is not None and raw != "" else int(default)
    except (TypeError, ValueError):
        value = int(default)
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def float_env(
    name: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    raw = os.environ.get(name)
    try:
        value = float(raw) if raw is not None and raw != "" else float(default)
    except (TypeError, ValueError):
        value = float(default)
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


ALLOWED_HOSTS = csv_env("ALLOWED_HOSTS", DEFAULT_ALLOWED_HOSTS)

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "aso",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "core.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "core.context_processors.version",
            ],
        },
    },
]

WSGI_APPLICATION = "core.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": DATA_DIR / "db.sqlite3",
    }
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

# CSRF trusted origins can be set from env (comma-separated).
DEFAULT_CSRF_TRUSTED_ORIGINS = [
    "http://localhost",
    "http://127.0.0.1",
    "http://respectaso.private",
    "http://localhost:9090",
    "http://127.0.0.1:9090",
    "http://respectaso.private:9090",
]
CSRF_TRUSTED_ORIGINS = csv_env(
    "CSRF_TRUSTED_ORIGINS",
    DEFAULT_CSRF_TRUSTED_ORIGINS,
)

RESULT_RETENTION_DAYS = int_env("RESULT_RETENTION_DAYS", 365, min_value=30, max_value=3650)
AUTO_REFRESH_MODE = os.environ.get("AUTO_REFRESH_MODE", "external").strip().lower()
if AUTO_REFRESH_MODE not in {"external", "thread"}:
    AUTO_REFRESH_MODE = "external"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_OPENAI_AVAILABLE_MODELS = [
    "gpt-5.2",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5",
    "gpt-4.1-mini",
    "gpt-4.1",
]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL
OPENAI_AVAILABLE_MODELS = []
for _model in csv_env("OPENAI_AVAILABLE_MODELS", DEFAULT_OPENAI_AVAILABLE_MODELS):
    _value = _model.strip()
    if _value and _value not in OPENAI_AVAILABLE_MODELS:
        OPENAI_AVAILABLE_MODELS.append(_value)
if OPENAI_MODEL not in OPENAI_AVAILABLE_MODELS:
    OPENAI_AVAILABLE_MODELS.insert(0, OPENAI_MODEL)
OPENAI_TIMEOUT_SECONDS = float_env("OPENAI_TIMEOUT_SECONDS", 40.0, min_value=5.0, max_value=300.0)
OPENAI_MAX_RETRIES = int_env("OPENAI_MAX_RETRIES", 2, min_value=0, max_value=10)
AI_MAX_CANDIDATES = int_env("AI_MAX_CANDIDATES", 12, min_value=3, max_value=50)
AI_EVALUATED_CANDIDATES = int_env("AI_EVALUATED_CANDIDATES", 8, min_value=1, max_value=30)
AI_MAX_COUNTRIES = int_env("AI_MAX_COUNTRIES", 3, min_value=1, max_value=10)
AI_HISTORY_ROWS_MAX = int_env("AI_HISTORY_ROWS_MAX", 300, min_value=50, max_value=5000)
AI_ENABLE_ONLINE_CONTEXT = bool_env("AI_ENABLE_ONLINE_CONTEXT", True)
AI_ONLINE_TOP_APPS_PER_COUNTRY = int_env(
    "AI_ONLINE_TOP_APPS_PER_COUNTRY",
    20,
    min_value=5,
    max_value=50,
)
DEFAULT_AI_SYSTEM_PROMPT = (
    "You are a senior Apple App Store Optimization analyst. "
    "Your job is to propose practical keyword opportunities for ONE specific iOS app. "
    "Always reason from the provided app metadata, tracked keyword history, and online market context. "
    "Prioritize discoverability + conversion intent + realistic competitiveness. "
    "Do not return duplicates, irrelevant terms, brand-infringing terms, or generic filler."
)
AI_SYSTEM_PROMPT = os.environ.get("AI_SYSTEM_PROMPT", DEFAULT_AI_SYSTEM_PROMPT).strip() or DEFAULT_AI_SYSTEM_PROMPT
DEFAULT_AI_USER_PROMPT_TEMPLATE = (
    "Task: propose high-quality Apple App Store keyword candidates for this app.\n"
    "Use ALL context below before deciding.\n"
    "For each candidate, the rationale must mention market or historical evidence.\n"
    "Prefer terms that match actual user intent for this app's use-case, not broad vanity terms.\n\n"
    "APP + HISTORICAL DATA JSON:\n"
    "{{SNAPSHOT_JSON}}\n\n"
    "ONLINE MARKET CONTEXT JSON:\n"
    "{{ONLINE_CONTEXT_JSON}}\n\n"
    "Return only valid JSON matching the provided schema."
)
AI_USER_PROMPT_TEMPLATE = (
    os.environ.get("AI_USER_PROMPT_TEMPLATE", DEFAULT_AI_USER_PROMPT_TEMPLATE).strip()
    or DEFAULT_AI_USER_PROMPT_TEMPLATE
)
