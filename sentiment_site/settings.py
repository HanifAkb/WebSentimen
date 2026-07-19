import importlib
import importlib.util
import os
from pathlib import Path

from django.contrib.messages import constants as message_constants
from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_list(name: str) -> list[str]:
    raw_value = os.getenv(name, "")
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return Path(value).expanduser()


DEFAULT_SECRET_KEY = "django-insecure-change-this-key"
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", DEFAULT_SECRET_KEY)
DEBUG = _env_bool("DJANGO_DEBUG", True)
ALLOWED_HOSTS: list[str] = _env_list("DJANGO_ALLOWED_HOSTS")
RAILWAY_PUBLIC_DOMAIN = (os.getenv("RAILWAY_PUBLIC_DOMAIN") or "").strip()

if RAILWAY_PUBLIC_DOMAIN and RAILWAY_PUBLIC_DOMAIN not in ALLOWED_HOSTS:
    ALLOWED_HOSTS.append(RAILWAY_PUBLIC_DOMAIN)

if not DEBUG and SECRET_KEY == DEFAULT_SECRET_KEY:
    raise ImproperlyConfigured("Set DJANGO_SECRET_KEY for production.")

if not DEBUG and not ALLOWED_HOSTS:
    raise ImproperlyConfigured("Set DJANGO_ALLOWED_HOSTS for production.")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "sentiment_app",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

_HAS_WHITENOISE = importlib.util.find_spec("whitenoise") is not None

if _HAS_WHITENOISE:
    MIDDLEWARE.insert(1, "whitenoise.middleware.WhiteNoiseMiddleware")

ROOT_URLCONF = "sentiment_site.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    }
]

WSGI_APPLICATION = "sentiment_site.wsgi.application"
ASGI_APPLICATION = "sentiment_site.asgi.application"

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    try:
        db_url_module = importlib.import_module("dj_database_url")
    except ModuleNotFoundError as exc:
        raise ImproperlyConfigured(
            "DATABASE_URL terdeteksi, tapi paket 'dj-database-url' belum terpasang."
        ) from exc
    DATABASES = {
        "default": db_url_module.parse(
            DATABASE_URL,
            conn_max_age=600,
            ssl_require=not DEBUG,
        )
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# Username disimpan sebagai data biasa, password disimpan hash satu arah.
PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.PBKDF2PasswordHasher",
    "django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher",
    "django.contrib.auth.hashers.ScryptPasswordHasher",
]

LANGUAGE_CODE = "id"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

if _HAS_WHITENOISE and not DEBUG:
    STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
        },
        "staticfiles": {
            "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
        },
    }

PERSISTENT_STORAGE_ROOT = os.getenv("PERSISTENT_STORAGE_ROOT") or os.getenv("RAILWAY_VOLUME_MOUNT_PATH") or ""
_persistent_storage_root = Path(PERSISTENT_STORAGE_ROOT).expanduser() if PERSISTENT_STORAGE_ROOT.strip() else None

MEDIA_URL = "/media/"
MEDIA_ROOT = _env_path(
    "DJANGO_MEDIA_ROOT",
    (_persistent_storage_root / "media") if _persistent_storage_root is not None else (BASE_DIR / "media"),
)

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

SENTIMENT_UPLOAD_MAX_SIZE = _env_int("SENTIMENT_UPLOAD_MAX_SIZE", 10 * 1024 * 1024)
SENTIMENT_ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".txt"}
SENTIMENT_MODELS_DIR = _env_path(
    "SENTIMENT_MODELS_DIR",
    (_persistent_storage_root / "model_assets")
    if _persistent_storage_root is not None
    else (BASE_DIR / "sentiment_site" / "models"),
)
SENTIMENT_TWITTER_MAX_TOTAL_TWEETS = _env_int("SENTIMENT_TWITTER_MAX_TOTAL_TWEETS", 4000)
SENTIMENT_TWITTER_MAX_TWEETS_PER_WINDOW = _env_int("SENTIMENT_TWITTER_MAX_TWEETS_PER_WINDOW", 500)
SENTIMENT_TWITTER_MIN_TWEETS_PER_WINDOW = _env_int("SENTIMENT_TWITTER_MIN_TWEETS_PER_WINDOW", 80)
SENTIMENT_TWITTER_PREDICT_CHUNK_SIZE = _env_int("SENTIMENT_TWITTER_PREDICT_CHUNK_SIZE", 300)
SENTIMENT_TWITTER_TEMP_DB_THRESHOLD_DAYS = _env_int("SENTIMENT_TWITTER_TEMP_DB_THRESHOLD_DAYS", 90)
SENTIMENT_TWITTER_MAX_RUNTIME_SECONDS = _env_int("SENTIMENT_TWITTER_MAX_RUNTIME_SECONDS", 90)
SENTIMENT_WORDCLOUD_MAX_TEXTS_PER_LABEL = _env_int("SENTIMENT_WORDCLOUD_MAX_TEXTS_PER_LABEL", 1200)
SENTIMENT_WORDCLOUD_MAX_CHARS_PER_LABEL = _env_int("SENTIMENT_WORDCLOUD_MAX_CHARS_PER_LABEL", 160000)
SENTIMENT_WORDCLOUD_MAX_ROWS = _env_int("SENTIMENT_WORDCLOUD_MAX_ROWS", 1500)

MESSAGE_TAGS = {
    message_constants.DEBUG: "secondary",
    message_constants.INFO: "info",
    message_constants.SUCCESS: "success",
    message_constants.WARNING: "warning",
    message_constants.ERROR: "danger",
}

LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "home"
LOGOUT_REDIRECT_URL = "login"

# Security hardening (production-safe defaults, dev-friendly when DEBUG=True)
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_SECURE = _env_bool("DJANGO_SESSION_COOKIE_SECURE", not DEBUG)
SESSION_COOKIE_AGE = _env_int("DJANGO_SESSION_COOKIE_AGE", 60 * 60 * 12)

CSRF_COOKIE_HTTPONLY = True
CSRF_COOKIE_SAMESITE = "Lax"
CSRF_COOKIE_SECURE = _env_bool("DJANGO_CSRF_COOKIE_SECURE", not DEBUG)
CSRF_TRUSTED_ORIGINS = _env_list("DJANGO_CSRF_TRUSTED_ORIGINS")
if RAILWAY_PUBLIC_DOMAIN:
    railway_csrf_origin = f"https://{RAILWAY_PUBLIC_DOMAIN}"
    if railway_csrf_origin not in CSRF_TRUSTED_ORIGINS:
        CSRF_TRUSTED_ORIGINS.append(railway_csrf_origin)

SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"
SECURE_CROSS_ORIGIN_OPENER_POLICY = "same-origin"
X_FRAME_OPTIONS = "DENY"

SECURE_SSL_REDIRECT = _env_bool("DJANGO_SECURE_SSL_REDIRECT", not DEBUG)
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SECURE_HSTS_SECONDS = _env_int("DJANGO_SECURE_HSTS_SECONDS", 31536000 if not DEBUG else 0)
SECURE_HSTS_INCLUDE_SUBDOMAINS = _env_bool("DJANGO_SECURE_HSTS_INCLUDE_SUBDOMAINS", not DEBUG)
SECURE_HSTS_PRELOAD = _env_bool("DJANGO_SECURE_HSTS_PRELOAD", not DEBUG)
