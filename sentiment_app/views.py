from __future__ import annotations

import base64
import csv
import hashlib
import io
import random
import re
import threading
import time
import zipfile
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from django.conf import settings
from django.contrib.auth import login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.core.exceptions import PermissionDenied
from django.core.files.base import File
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.paginator import Paginator
from django.db import close_old_connections, models, transaction
from django.http import Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.urls import reverse
from django.views.decorators.http import require_POST

from .forms import (
    AdminCreateUserForm,
    AdminModelEditForm,
    AdminModelUploadForm,
    AdminEditUserForm,
    AdminPredictionHistoryForm,
    AdminScrapeHistoryForm,
    LoginForm,
    PredictForm,
    ResumeScrapeForm,
    TwitterFetchForm,
)
from .models import (
    PredictionHistory,
    ScrapeHistory,
    ScrapeTempChunk,
    SentimentModelVersion,
    sentiment_model_storage_name,
)
from .services.file_service import (
    FileValidationError,
    parse_uploaded_file,
)
from .services.model_service import (
    ModelServiceError,
    clear_cache,
    predict_batch,
    predict_batch_in_chunks,
    predict_single,
    preprocess_text,
    resolve_model_version_name,
)
from .services.scraping_service import TwitterAPIError, TwitterRateLimitError, TwitterTimeoutError, fetch_tweets

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

DEFAULT_PER_PAGE = 10
MAX_PER_PAGE = 200
HISTORY_PER_PAGE = 10
TWITTER_RESULT_SESSION_KEY = "twitter_last_result"
PREDICTION_LABEL_COLUMNS = ["knn_label", "svm_label", "combined_label"]
PREDICTION_SCORE_COLUMNS = [
    "knn_positive_score",
    "knn_negative_score",
    "svm_positive_score",
    "svm_negative_score",
    "combined_positive_score",
    "combined_negative_score",
]
PREDICTION_COLUMNS = [
    "knn_positive_score",
    "knn_negative_score",
    "svm_positive_score",
    "svm_negative_score",
    "combined_positive_score",
    "combined_negative_score",
    "knn_label",
    "svm_label",
    "combined_label",
]
LEGACY_PREDICTION_COLUMNS = {"knn_score", "svm_score", "combined_score"}
ALL_PREDICTION_COLUMNS = set(PREDICTION_COLUMNS) | LEGACY_PREDICTION_COLUMNS
BACKGROUND_HISTORY_RETRY_LIMIT = 6
BACKGROUND_HISTORY_RETRY_DELAY_SECONDS = 8
PREDICTION_HEADERS = {
    "knn_label": "KNN",
    "knn_positive_score": "Probabilitas Positif KNN",
    "knn_negative_score": "Probabilitas Negatif KNN",
    "svm_label": "SVM",
    "svm_positive_score": "Probabilitas Positif SVM",
    "svm_negative_score": "Probabilitas Negatif SVM",
    "combined_label": "Gabungan (Soft Voting)",
    "combined_positive_score": "Probabilitas Positif Soft Voting",
    "combined_negative_score": "Probabilitas Negatif Soft Voting",
}
PREDICTION_TABLE_HEADERS = {
    "knn_label": "KNN",
    "knn_positive_score": "Positif (KNN)",
    "knn_negative_score": "Negatif (KNN)",
    "svm_label": "SVM",
    "svm_positive_score": "Positif (SVM)",
    "svm_negative_score": "Negatif (SVM)",
    "combined_label": "Gabungan (Soft Voting)",
    "combined_positive_score": "Positif (Soft Voting)",
    "combined_negative_score": "Negatif (Soft Voting)",
}
SCRAPE_BASE_COLUMNS = [
    "id",
    "url",
    "text",
    "retweetCount",
    "replyCount",
    "likeCount",
    "quoteCount",
    "viewCount",
    "CreatedAt",
    "lang",
    "bookmarkCount",
    "isReply",
    "inReplyTold",
    "userName",
    "image_tweet",
]
SCRAPE_EXPORT_COLUMNS = SCRAPE_BASE_COLUMNS + PREDICTION_COLUMNS
SCRAPE_HEADERS = {
    "id": "ID",
    "url": "URL",
    "text": "Teks",
    "retweetCount": "Retweet",
    "replyCount": "Reply",
    "likeCount": "Like",
    "quoteCount": "Quote",
    "viewCount": "View",
    "CreatedAt": "CreatedAt",
    "lang": "Lang",
    "bookmarkCount": "Bookmark",
    "isReply": "IsReply",
    "inReplyTold": "inReplyTold",
    "userName": "User Name",
    "image_tweet": "Gambar Tweet",
    **PREDICTION_HEADERS,
}


def _launch_background_history_job(
    job_name: str,
    target,
    *args,
    **kwargs,
) -> None:
    def _run() -> None:
        close_old_connections()
        try:
            target(*args, **kwargs)
        finally:
            close_old_connections()

    def _start() -> None:
        worker = threading.Thread(target=_run, name=job_name, daemon=True)
        worker.start()

    transaction.on_commit(_start)
PREDICTION_DATE_COLUMN_HINTS = (
    "createdat",
    "created_at",
    "created",
    "date",
    "datetime",
    "time",
    "tanggal",
    "waktu",
)
_WORDCLOUD_STOPWORDS_CACHE: set[str] | None = None
_WORDCLOUD_STOPWORDS_CACHE_SIGNATURE: tuple[tuple[str, bool, int | None, int | None], ...] | None = None
WORDCLOUD_POSITIVE_COLORS = ["#14532d", "#166534", "#15803d", "#16a34a", "#22c55e"]
WORDCLOUD_NEGATIVE_COLORS = ["#7f1d1d", "#991b1b", "#b91c1c", "#dc2626", "#ef4444"]
WORDCLOUD_STOPWORDS_PATHS: list[Path] | None = None
INDONESIAN_MONTH_NAMES = (
    "",
    "Januari",
    "Februari",
    "Maret",
    "April",
    "Mei",
    "Juni",
    "Juli",
    "Agustus",
    "September",
    "Oktober",
    "November",
    "Desember",
)


def _wordcloud_stopwords_paths() -> list[Path]:
    if WORDCLOUD_STOPWORDS_PATHS is not None:
        return list(WORDCLOUD_STOPWORDS_PATHS)

    models_dir = Path(getattr(settings, "SENTIMENT_MODELS_DIR", settings.BASE_DIR / "sentiment_site" / "models"))
    return [
        models_dir / "stopwords-id(wordcloud).txt",
        models_dir / "stopwords-id.txt",
        Path(settings.BASE_DIR) / "sentiment_site" / "models" / "stopwords-id(wordcloud).txt",
        Path(settings.BASE_DIR) / "sentiment_site" / "models" / "stopwords-id.txt",
    ]


def _setting_positive_int(name: str, default: int) -> int:
    raw_value = getattr(settings, name, default)
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _safe_positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _normalize_per_page(value: object, default: int = DEFAULT_PER_PAGE) -> int:
    parsed = _safe_positive_int(value, default)
    return min(parsed, MAX_PER_PAGE)


def _is_truthy_flag(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _build_url_with_dashboard_flag(request: HttpRequest) -> str:
    query_params = request.GET.copy()
    query_params["dashboard"] = "1"
    encoded = query_params.urlencode()
    return f"{request.path}?{encoded}" if encoded else request.path


def _paginate_rows(
    rows: list[dict[str, object]],
    page: int,
    per_page: int,
) -> tuple[list[dict[str, object]], int, int, int]:
    total_rows = len(rows)
    total_pages = max(1, (total_rows + per_page - 1) // per_page) if total_rows else 1
    current_page = max(1, min(page, total_pages))

    if total_rows == 0:
        return [], 0, current_page, total_pages

    start_idx = (current_page - 1) * per_page
    end_idx = start_idx + per_page
    page_slice = rows[start_idx:end_idx]

    return _build_paginated_rows(page_slice, total_rows, current_page, total_pages, start_idx)


def _build_paginated_rows(
    page_slice: list[dict[str, object]],
    total_rows: int,
    current_page: int,
    total_pages: int,
    start_idx: int,
) -> tuple[list[dict[str, object]], int, int, int]:
    if total_rows == 0:
        return [], 0, current_page, total_pages

    page_rows: list[dict[str, object]] = []
    for idx, row in enumerate(page_slice, start=1):
        normalized_row = dict(row)
        normalized_row["row_number"] = start_idx + idx
        page_rows.append(normalized_row)

    return page_rows, total_rows, current_page, total_pages


def _paginate_scrape_history_rows(
    history: ScrapeHistory,
    page: int,
    per_page: int,
) -> tuple[list[dict[str, object]], int, int, int]:
    base_rows = history.rows if isinstance(history.rows, list) else []
    total_rows = max(_safe_positive_int(history.tweet_count, 0), len(base_rows))

    if total_rows <= len(base_rows):
        computed_total_rows = len(base_rows)
        for chunk_rows in history.temp_chunks.order_by("chunk_index", "id").values_list("rows", flat=True):
            if isinstance(chunk_rows, list):
                computed_total_rows += len(chunk_rows)
        total_rows = computed_total_rows

    if total_rows <= len(base_rows):
        return _paginate_rows(base_rows, page, per_page)

    total_pages = max(1, (total_rows + per_page - 1) // per_page)
    current_page = max(1, min(page, total_pages))
    start_idx = (current_page - 1) * per_page
    end_idx = start_idx + per_page
    page_slice: list[dict[str, object]] = []
    offset = 0

    def extend_page_rows(source_rows: list[dict[str, object]]) -> bool:
        nonlocal offset, page_slice
        source_count = len(source_rows)
        if source_count <= 0:
            return offset >= end_idx

        slice_start = max(0, start_idx - offset)
        slice_end = min(source_count, end_idx - offset)
        if slice_start < slice_end:
            page_slice.extend(source_rows[slice_start:slice_end])
        offset += source_count
        return offset >= end_idx

    if extend_page_rows(base_rows):
        return _build_paginated_rows(page_slice, total_rows, current_page, total_pages, start_idx)

    for chunk_rows in history.temp_chunks.order_by("chunk_index", "id").values_list("rows", flat=True):
        if isinstance(chunk_rows, list) and extend_page_rows(chunk_rows):
            break

    return _build_paginated_rows(page_slice, total_rows, current_page, total_pages, start_idx)


def _build_batch_preview(
    source_rows: list[dict[str, str]],
    source_columns: list[str],
    predictions: list[dict[str, object]],
) -> tuple[list[str], list[dict[str, object]]]:
    columns = [column for column in source_columns if str(column).strip().lower() != "id"] + PREDICTION_COLUMNS
    headers = [PREDICTION_TABLE_HEADERS.get(column, column) for column in columns]

    preview_rows: list[dict[str, object]] = []
    for index, (source_row, prediction) in enumerate(zip(source_rows, predictions), start=1):
        merged = {
            **source_row,
            "knn_label": prediction.get("knn_label", ""),
            "knn_positive_score": prediction.get("knn_positive_score"),
            "knn_negative_score": prediction.get("knn_negative_score"),
            "svm_label": prediction.get("svm_label", ""),
            "svm_positive_score": prediction.get("svm_positive_score"),
            "svm_negative_score": prediction.get("svm_negative_score"),
            "combined_label": prediction.get("combined_label", ""),
            "combined_positive_score": prediction.get("combined_positive_score"),
            "combined_negative_score": prediction.get("combined_negative_score"),
        }

        cells = []
        for column in columns:
            cell_type = "text"
            if column in PREDICTION_LABEL_COLUMNS:
                cell_type = "label"
            elif column in PREDICTION_SCORE_COLUMNS:
                cell_type = "score"

            cells.append(
                {
                    "type": cell_type,
                    "value": merged.get(column, ""),
                }
            )

        preview_rows.append({"index": index, "cells": cells})

    return headers, preview_rows


def _merge_batch_rows_for_history(
    source_rows: list[dict[str, str]],
    predictions: list[dict[str, object]],
) -> list[dict[str, object]]:
    merged_rows: list[dict[str, object]] = []
    for source_row, prediction in zip(source_rows, predictions):
        merged_rows.append(
            {
                **source_row,
                "knn_label": prediction.get("knn_label", ""),
                "knn_positive_score": prediction.get("knn_positive_score"),
                "knn_negative_score": prediction.get("knn_negative_score"),
                "svm_label": prediction.get("svm_label", ""),
                "svm_positive_score": prediction.get("svm_positive_score"),
                "svm_negative_score": prediction.get("svm_negative_score"),
                "combined_label": prediction.get("combined_label", ""),
                "combined_positive_score": prediction.get("combined_positive_score"),
                "combined_negative_score": prediction.get("combined_negative_score"),
            }
        )
    return merged_rows


def _normalize_prediction_source_columns(
    columns_value: object,
    rows: list[dict[str, object]],
) -> list[str]:
    columns: list[str] = []
    if isinstance(columns_value, list):
        columns = [str(column) for column in columns_value if isinstance(column, str)]

    if columns:
        return columns

    if not rows:
        return []

    first_row = rows[0]
    derived_columns: list[str] = []
    for key in first_row.keys():
        if key not in ALL_PREDICTION_COLUMNS and key != "row_number":
            derived_columns.append(str(key))
    return derived_columns


def _build_prediction_history_preview(
    rows: list[dict[str, object]],
    source_columns: list[str],
) -> tuple[list[str], list[dict[str, object]]]:
    columns = [column for column in source_columns if str(column).strip().lower() != "id"] + PREDICTION_COLUMNS
    headers = [PREDICTION_TABLE_HEADERS.get(column, column) for column in columns]
    preview_rows: list[dict[str, object]] = []

    for row in rows:
        index = int(row.get("row_number", len(preview_rows) + 1))
        cells: list[dict[str, object]] = []
        for column in columns:
            cell_type = "text"
            if column in PREDICTION_LABEL_COLUMNS:
                cell_type = "label"
            elif column in PREDICTION_SCORE_COLUMNS:
                cell_type = "score"
            cells.append(
                {
                    "type": cell_type,
                    "value": row.get(column, ""),
                }
            )
        preview_rows.append({"index": index, "cells": cells})

    return headers, preview_rows


def _resolve_prediction_text_column(
    text_column: str,
    source_columns: list[str],
) -> str:
    preferred = str(text_column or "").strip()
    if preferred:
        return preferred

    for column in source_columns:
        normalized = str(column).strip().lower()
        if normalized in {"text", "tweet", "content", "sentence", "review", "kalimat"}:
            return str(column)

    return str(source_columns[0]) if source_columns else ""


def _safe_parse_iso_date(value: object) -> date | None:
    if value in (None, ""):
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _parse_created_at_date(value: object) -> date | None:
    if value in (None, ""):
        return None

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), timezone.utc).date()
        except (OverflowError, OSError, ValueError):
            return None

    raw = str(value).strip()
    if not raw:
        return None

    if raw.isdigit():
        try:
            timestamp = float(raw)
            if timestamp > 10_000_000_000:
                timestamp = timestamp / 1000.0
            return datetime.fromtimestamp(timestamp, timezone.utc).date()
        except (OverflowError, OSError, ValueError):
            return None

    normalized = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        pass

    candidate_formats = (
        "%a %b %d %H:%M:%S %z %Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%d",
    )
    for fmt in candidate_formats:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _bucket_start(value: date, granularity: str, anchor: date | None = None) -> date:
    if granularity == "week":
        # Weekly buckets are anchored to requested start_date so chart range
        # does not spill into dates before user input.
        anchor_date = anchor or value
        delta_days = (value - anchor_date).days
        return anchor_date + timedelta(days=(delta_days // 7) * 7)
    if granularity == "month":
        return value.replace(day=1)
    return value


def _bucket_end(value: date, granularity: str) -> date:
    if granularity == "day":
        return value
    if granularity == "week":
        return value + timedelta(days=6)
    return _next_bucket(value, "month") - timedelta(days=1)


def _next_bucket(value: date, granularity: str) -> date:
    if granularity == "day":
        return value + timedelta(days=1)
    if granularity == "week":
        return value + timedelta(days=7)
    year = value.year + (1 if value.month == 12 else 0)
    month = 1 if value.month == 12 else value.month + 1
    return date(year, month, 1)


def _format_bucket_label(
    value: date,
    granularity: str,
    range_start: date | None = None,
    range_end: date | None = None,
) -> str:
    display_start = value
    display_end = _bucket_end(value, granularity)
    if range_start and display_start < range_start:
        display_start = range_start
    if range_end and display_end > range_end:
        display_end = range_end

    if granularity == "day":
        return display_start.strftime("%d %b %Y")
    if granularity == "week":
        return f"{display_start.strftime('%d %b')} - {display_end.strftime('%d %b %Y')}"

    # Keep concise month label only when bucket covers a full calendar month.
    full_month_start = value.replace(day=1)
    full_month_end = _bucket_end(full_month_start, "month")
    if display_start == full_month_start and display_end == full_month_end:
        return value.strftime("%b %Y")
    return f"{display_start.strftime('%d %b')} - {display_end.strftime('%d %b %Y')}"


def _normalize_sentiment_label(value: object) -> str:
    text = str(value or "").strip().lower()
    if text == "positive":
        return "Positive"
    if text == "negative":
        return "Negative"
    if text == "neutral":
        return "Neutral"
    return "Neutral"


def _display_sentiment_label_id(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"positive", "positif"}:
        return "Positif"
    if normalized in {"negative", "negatif"}:
        return "Negatif"
    if normalized in {"neutral", "netral"}:
        return "Netral"
    return str(value or "")


def _build_combined_sentiment_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, dict):
            continue
        counts[_normalize_sentiment_label(row.get("combined_label"))] += 1
    return {
        "positive": int(counts.get("Positive", 0)),
        "neutral": int(counts.get("Neutral", 0)),
        "negative": int(counts.get("Negative", 0)),
    }


def _build_scrape_history_combined_counts(history: ScrapeHistory) -> dict[str, int]:
    counts: Counter[str] = Counter()

    def add_rows(source_rows: object) -> None:
        if not isinstance(source_rows, list):
            return
        for row in source_rows:
            if not isinstance(row, dict):
                continue
            counts[_normalize_sentiment_label(row.get("combined_label"))] += 1

    add_rows(history.rows)
    for chunk_rows in history.temp_chunks.order_by("chunk_index", "id").values_list("rows", flat=True):
        add_rows(chunk_rows)

    return {
        "positive": int(counts.get("Positive", 0)),
        "neutral": int(counts.get("Neutral", 0)),
        "negative": int(counts.get("Negative", 0)),
    }


def _load_wordcloud_stopwords() -> set[str]:
    global _WORDCLOUD_STOPWORDS_CACHE, _WORDCLOUD_STOPWORDS_CACHE_SIGNATURE

    stopword_paths = _wordcloud_stopwords_paths()
    current_signature: list[tuple[str, bool, int | None, int | None, str | None]] = []
    for path in stopword_paths:
        try:
            stat_result = path.stat()
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            current_signature.append(
                (str(path), True, stat_result.st_mtime_ns, stat_result.st_size, content_hash)
            )
        except OSError:
            current_signature.append((str(path), False, None, None, None))

    signature_tuple = tuple(current_signature)
    if (
        _WORDCLOUD_STOPWORDS_CACHE is not None
        and _WORDCLOUD_STOPWORDS_CACHE_SIGNATURE == signature_tuple
    ):
        return _WORDCLOUD_STOPWORDS_CACHE

    for path in stopword_paths:
        if not path.exists():
            continue
        try:
            loaded = {
                line.strip().lower()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            if loaded:
                _WORDCLOUD_STOPWORDS_CACHE = loaded
                _WORDCLOUD_STOPWORDS_CACHE_SIGNATURE = signature_tuple
                return _WORDCLOUD_STOPWORDS_CACHE
        except Exception:
            continue

    _WORDCLOUD_STOPWORDS_CACHE = set()
    _WORDCLOUD_STOPWORDS_CACHE_SIGNATURE = signature_tuple
    return _WORDCLOUD_STOPWORDS_CACHE


def _clean_text_for_wordcloud(text: str) -> str:
    # Align with model preprocessing so WordCloud reflects the same cleaned language space.
    cleaned = preprocess_text(str(text or ""), apply_stemming=True)
    cleaned = re.sub(r"[^0-9a-zA-Z_\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _wordcloud_excluded_terms(query: str | None = None) -> set[str]:
    cleaned_query = _clean_text_for_wordcloud(str(query or ""))
    if not cleaned_query:
        return set()
    return {token for token in cleaned_query.split() if token}


def _palette_color_func(colors: list[str]):
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        picker = random_state if random_state is not None else random
        return colors[picker.randint(0, len(colors) - 1)]

    return color_func


def _build_wordcloud_image(
    texts: list[str],
    colormap: str | list[str],
    excluded_terms: set[str] | None = None,
) -> str | None:
    unigram_counter = _build_wordcloud_unigram_counter(texts, excluded_terms=excluded_terms)
    if WordCloud is None or not unigram_counter:
        return None

    use_palette = isinstance(colormap, list)
    cloud = WordCloud(
        width=1400,
        height=800,
        background_color="white",
        stopwords=_load_wordcloud_stopwords(),
        collocations=False,
        max_words=140,
        margin=24,
        min_font_size=10,
        max_font_size=96,
        relative_scaling=0.35,
        prefer_horizontal=1.0,
        color_func=_palette_color_func(colormap) if use_palette else None,
        colormap=None if use_palette else colormap,
    ).generate_from_frequencies(unigram_counter)

    buffer = io.BytesIO()
    cloud.to_image().save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _build_wordcloud_unigram_counter(
    texts: list[str],
    excluded_terms: set[str] | None = None,
) -> Counter[str]:
    unigram_counter: Counter[str] = Counter()
    wordcloud_stopwords = _load_wordcloud_stopwords()
    blocked_terms = set(excluded_terms or set())
    for raw_text in texts:
        cleaned_text = _clean_text_for_wordcloud(str(raw_text or ""))
        if not cleaned_text:
            continue
        tokens = [
            token
            for token in cleaned_text.split()
            if token and token not in wordcloud_stopwords and token not in blocked_terms
        ]
        if not tokens:
            continue
        unigram_counter.update(tokens)
    return unigram_counter


def _top_unigram_stats(unigram_counter: Counter[str], limit: int = 5) -> list[dict[str, object]]:
    if not unigram_counter:
        return []
    return [
        {
            "word": word,
            "count": int(count),
        }
        for word, count in unigram_counter.most_common(max(1, int(limit)))
    ]


def _choose_granularity(start_date: date, end_date: date) -> tuple[str, str]:
    total_days = max(1, (end_date - start_date).days)
    if total_days <= 14:
        return "day", "Harian"
    if total_days < 60:
        return "week", "Mingguan"
    if total_days <= 61:
        return "week", "Mingguan"
    return "month", "Bulanan"


def _format_indonesian_date(value: date) -> str:
    month_name = INDONESIAN_MONTH_NAMES[value.month]
    return f"{value.day:02d} {month_name} {value.year}"


def _build_scraping_dashboard(
    rows: list[dict[str, object]],
    start_date: date | None = None,
    end_date: date | None = None,
    trend_subject: str = "Tweet",
    query: str | None = None,
) -> dict[str, object]:
    max_texts_per_label = _setting_positive_int("SENTIMENT_WORDCLOUD_MAX_TEXTS_PER_LABEL", 1200)
    max_chars_per_label = _setting_positive_int("SENTIMENT_WORDCLOUD_MAX_CHARS_PER_LABEL", 160000)

    combined_positive_texts: list[str] = []
    combined_negative_texts: list[str] = []
    combined_positive_chars = 0
    combined_negative_chars = 0
    knn_counts: Counter[str] = Counter()
    svm_counts: Counter[str] = Counter()
    combined_counts: Counter[str] = Counter()
    row_dates: list[date] = []

    for row in rows:
        text = str(row.get("text", "") or "")

        knn_label = _normalize_sentiment_label(row.get("knn_label"))
        svm_label = _normalize_sentiment_label(row.get("svm_label"))
        combined_label = _normalize_sentiment_label(row.get("combined_label"))
        knn_counts[knn_label] += 1
        svm_counts[svm_label] += 1
        combined_counts[combined_label] += 1

        if text:
            if combined_label == "Positive":
                if len(combined_positive_texts) < max_texts_per_label and combined_positive_chars < max_chars_per_label:
                    clipped_text = text[: max(1, max_chars_per_label - combined_positive_chars)]
                    combined_positive_chars += len(clipped_text)
                    combined_positive_texts.append(clipped_text)
            elif combined_label == "Negative":
                if len(combined_negative_texts) < max_texts_per_label and combined_negative_chars < max_chars_per_label:
                    clipped_text = text[: max(1, max_chars_per_label - combined_negative_chars)]
                    combined_negative_chars += len(clipped_text)
                    combined_negative_texts.append(clipped_text)

        created_date = _parse_created_at_date(row.get("CreatedAt"))
        if created_date is None:
            created_date = _parse_created_at_date(row.get("_week_start"))
        if created_date is not None:
            row_dates.append(created_date)

    if start_date is None:
        start_date = min(row_dates) if row_dates else date.today()
    if end_date is None:
        end_date = max(row_dates) if row_dates else start_date
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    granularity, granularity_label = _choose_granularity(start_date, end_date)
    bucket_anchor = start_date if granularity == "week" else None
    trend_counter: Counter[date] = Counter()
    for created_date in row_dates:
        trend_counter[_bucket_start(created_date, granularity, bucket_anchor)] += 1

    chart_labels: list[str] = []
    chart_values: list[int] = []
    bucket_cursor = _bucket_start(start_date, granularity, bucket_anchor)
    bucket_last = _bucket_start(end_date, granularity, bucket_anchor)
    while bucket_cursor <= bucket_last:
        chart_labels.append(_format_bucket_label(bucket_cursor, granularity, start_date, end_date))
        chart_values.append(int(trend_counter.get(bucket_cursor, 0)))
        bucket_cursor = _next_bucket(bucket_cursor, granularity)

    wordcloud_error = ""
    max_wordcloud_rows = _setting_positive_int("SENTIMENT_WORDCLOUD_MAX_ROWS", 1500)
    wordclouds: dict[str, str | None] = {
        "combined_positive_image": None,
        "combined_negative_image": None,
    }
    wordcloud_top_unigrams: dict[str, list[dict[str, object]]] = {
        "combined_positive": [],
        "combined_negative": [],
    }
    if len(rows) > max_wordcloud_rows:
        wordcloud_error = (
            f"WordCloud dinonaktifkan otomatis untuk data lebih dari {max_wordcloud_rows} baris "
            "agar proses tetap stabil."
        )
    elif WordCloud is None:
        wordcloud_error = (
            "Library `wordcloud` belum terpasang. Jalankan `pip install wordcloud` "
            "lalu restart server untuk menampilkan WordCloud."
        )
    else:
        try:
            excluded_terms = _wordcloud_excluded_terms(query)
            positive_unigram_counter = _build_wordcloud_unigram_counter(
                combined_positive_texts,
                excluded_terms=excluded_terms,
            )
            negative_unigram_counter = _build_wordcloud_unigram_counter(
                combined_negative_texts,
                excluded_terms=excluded_terms,
            )
            wordclouds["combined_positive_image"] = _build_wordcloud_image(
                combined_positive_texts,
                colormap=WORDCLOUD_POSITIVE_COLORS,
                excluded_terms=excluded_terms,
            )
            wordclouds["combined_negative_image"] = _build_wordcloud_image(
                combined_negative_texts,
                colormap=WORDCLOUD_NEGATIVE_COLORS,
                excluded_terms=excluded_terms,
            )
            wordcloud_top_unigrams["combined_positive"] = _top_unigram_stats(positive_unigram_counter)
            wordcloud_top_unigrams["combined_negative"] = _top_unigram_stats(negative_unigram_counter)
        except Exception as exc:
            wordcloud_error = f"Gagal membuat WordCloud: {exc}"

    return {
        "period_label": f"{_format_indonesian_date(start_date)} - {_format_indonesian_date(end_date)}",
        "granularity_label": granularity_label,
        "knn_counts": {
            "positive": int(knn_counts.get("Positive", 0)),
            "negative": int(knn_counts.get("Negative", 0)),
            "neutral": int(knn_counts.get("Neutral", 0)),
        },
        "svm_counts": {
            "positive": int(svm_counts.get("Positive", 0)),
            "negative": int(svm_counts.get("Negative", 0)),
            "neutral": int(svm_counts.get("Neutral", 0)),
        },
        "combined_counts": {
            "positive": int(combined_counts.get("Positive", 0)),
            "negative": int(combined_counts.get("Negative", 0)),
            "neutral": int(combined_counts.get("Neutral", 0)),
        },
        "wordcloud_available": WordCloud is not None and not wordcloud_error,
        "wordcloud_error": wordcloud_error,
        "wordclouds": wordclouds,
        "wordcloud_top_unigrams": wordcloud_top_unigrams,
        "charts": {
            "pie_labels": ["Positif", "Negatif", "Netral"],
            "knn_pie": [
                int(knn_counts.get("Positive", 0)),
                int(knn_counts.get("Negative", 0)),
                int(knn_counts.get("Neutral", 0)),
            ],
            "svm_pie": [
                int(svm_counts.get("Positive", 0)),
                int(svm_counts.get("Negative", 0)),
                int(svm_counts.get("Neutral", 0)),
            ],
            "combined_pie": [
                int(combined_counts.get("Positive", 0)),
                int(combined_counts.get("Negative", 0)),
                int(combined_counts.get("Neutral", 0)),
            ],
            "trend_labels": chart_labels,
            "trend_values": chart_values,
            "trend_title": f"Jumlah {trend_subject} per {granularity_label}",
        },
    }


def _row_value_for_column(row: dict[str, object], column: str) -> object:
    if column in row:
        return row.get(column)

    normalized_column = column.strip().lower()
    for key, value in row.items():
        if str(key).strip().lower() == normalized_column:
            return value
    return ""


def _prediction_row_text(
    row: dict[str, object],
    text_column: str,
    source_columns: list[str],
) -> str:
    candidate_columns: list[str] = []
    if text_column:
        candidate_columns.append(text_column)
    candidate_columns.extend(["text", "tweet", "content", "sentence"])
    candidate_columns.extend(source_columns)

    seen_columns: set[str] = set()
    for column in candidate_columns:
        normalized_column = str(column or "").strip().lower()
        if not normalized_column or normalized_column in seen_columns:
            continue
        seen_columns.add(normalized_column)
        value = _row_value_for_column(row, str(column))
        text = str(value or "").strip()
        if text:
            return text

    for key, value in row.items():
        if key in ALL_PREDICTION_COLUMNS or key == "row_number":
            continue
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _prediction_row_date_value(row: dict[str, object]) -> object:
    for key, value in row.items():
        normalized_key = str(key).strip().lower()
        if normalized_key in PREDICTION_DATE_COLUMN_HINTS and _parse_created_at_date(value) is not None:
            return value

    for key, value in row.items():
        normalized_key = str(key).strip().lower()
        if key in ALL_PREDICTION_COLUMNS or key == "row_number":
            continue
        if not any(hint in normalized_key for hint in PREDICTION_DATE_COLUMN_HINTS):
            continue
        if _parse_created_at_date(value) is not None:
            return value

    return ""


def _rows_need_probability_split_upgrade(rows: list[dict[str, object]]) -> bool:
    required_columns = {
        "knn_positive_score",
        "knn_negative_score",
        "svm_positive_score",
        "svm_negative_score",
        "combined_label",
        "combined_positive_score",
        "combined_negative_score",
    }
    for row in rows:
        if not isinstance(row, dict):
            return True
        if not required_columns.issubset(set(row.keys())):
            return True
    return False


def _upgrade_prediction_history_scores_if_needed(history: PredictionHistory, *, force: bool = False) -> PredictionHistory:
    stored_rows = history.rows if isinstance(history.rows, list) else []
    if not force and not _rows_need_probability_split_upgrade([row for row in stored_rows if isinstance(row, dict)]):
        return history

    if not stored_rows:
        return history

    source_columns = _normalize_prediction_source_columns(
        history.columns,
        [row for row in stored_rows if isinstance(row, dict)],
    )
    normalized_rows: list[dict[str, object]] = []
    texts: list[str] = []
    for row in stored_rows:
        normalized_row = dict(row) if isinstance(row, dict) else {}
        normalized_rows.append(normalized_row)
        texts.append(_prediction_row_text(normalized_row, history.text_column, source_columns))

    predictions = predict_batch(texts)
    updated_rows: list[dict[str, object]] = []

    for row, prediction in zip(normalized_rows, predictions):
        updated_row = dict(row)
        updated_row["knn_label"] = prediction.get("knn_label", "")
        updated_row["knn_positive_score"] = prediction.get("knn_positive_score")
        updated_row["knn_negative_score"] = prediction.get("knn_negative_score")
        updated_row["svm_label"] = prediction.get("svm_label", "")
        updated_row["svm_positive_score"] = prediction.get("svm_positive_score")
        updated_row["svm_negative_score"] = prediction.get("svm_negative_score")
        updated_row["combined_label"] = prediction.get("combined_label", "")
        updated_row["combined_positive_score"] = prediction.get("combined_positive_score")
        updated_row["combined_negative_score"] = prediction.get("combined_negative_score")
        updated_row.pop("knn_score", None)
        updated_row.pop("svm_score", None)
        updated_row.pop("combined_score", None)
        updated_rows.append(updated_row)

    history.rows = _serialize_history_rows(updated_rows)
    history.save(update_fields=["rows"])
    return history


def _store_scrape_history_rows(history: ScrapeHistory, rows: list[dict[str, object]]) -> None:
    serialized_rows = _serialize_history_rows(rows)
    base_rows = history.rows if isinstance(history.rows, list) else []
    base_count = len(base_rows)
    chunk_objects = list(history.temp_chunks.order_by("chunk_index", "id"))

    if not chunk_objects:
        history.rows = serialized_rows
        history.save(update_fields=["rows"])
        return

    history.rows = serialized_rows[:base_count]
    history.save(update_fields=["rows"])

    offset = base_count
    for chunk in chunk_objects:
        current_chunk_rows = chunk.rows if isinstance(chunk.rows, list) else []
        chunk_count = len(current_chunk_rows)
        chunk.rows = serialized_rows[offset : offset + chunk_count]
        chunk.save(update_fields=["rows"])
        offset += chunk_count

    if offset < len(serialized_rows):
        history.rows = list(history.rows) + serialized_rows[offset:]
        history.save(update_fields=["rows"])


def _upgrade_scrape_history_scores_if_needed(history: ScrapeHistory, *, force: bool = False) -> ScrapeHistory:
    stored_rows = _load_scrape_rows(history)
    if not force and not _rows_need_probability_split_upgrade([row for row in stored_rows if isinstance(row, dict)]):
        return history

    if not stored_rows:
        return history

    normalized_rows: list[dict[str, object]] = []
    texts: list[str] = []
    for row in stored_rows:
        normalized_row = dict(row) if isinstance(row, dict) else {}
        normalized_rows.append(normalized_row)
        texts.append(str(normalized_row.get("text", "") or "").strip())

    predictions = predict_batch(texts)
    updated_rows: list[dict[str, object]] = []
    for row, prediction in zip(normalized_rows, predictions):
        updated_row = dict(row)
        updated_row["knn_label"] = prediction.get("knn_label", "")
        updated_row["knn_positive_score"] = prediction.get("knn_positive_score")
        updated_row["knn_negative_score"] = prediction.get("knn_negative_score")
        updated_row["svm_label"] = prediction.get("svm_label", "")
        updated_row["svm_positive_score"] = prediction.get("svm_positive_score")
        updated_row["svm_negative_score"] = prediction.get("svm_negative_score")
        updated_row["combined_label"] = prediction.get("combined_label", "")
        updated_row["combined_positive_score"] = prediction.get("combined_positive_score")
        updated_row["combined_negative_score"] = prediction.get("combined_negative_score")
        updated_row.pop("knn_score", None)
        updated_row.pop("svm_score", None)
        updated_row.pop("combined_score", None)
        updated_rows.append(updated_row)

    _store_scrape_history_rows(history, updated_rows)
    return history


def _history_created_date(history: PredictionHistory) -> date:
    created_at = getattr(history, "created_at", None)
    if isinstance(created_at, datetime):
        return created_at.date()
    return date.today()


def _build_prediction_dashboard(
    history: PredictionHistory,
    rows: list[dict[str, object]],
    source_columns: list[str],
) -> dict[str, object]:
    fallback_date = _history_created_date(history)
    fallback_date_text = fallback_date.isoformat()
    dashboard_rows: list[dict[str, object]] = []
    dashboard_dates: list[date] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        created_at_value = _prediction_row_date_value(row) or fallback_date_text
        created_date = _parse_created_at_date(created_at_value)
        if created_date is not None:
            dashboard_dates.append(created_date)

        dashboard_rows.append(
            {
                "text": _prediction_row_text(row, history.text_column, source_columns),
                "CreatedAt": created_at_value,
                "knn_label": row.get("knn_label", ""),
                "svm_label": row.get("svm_label", ""),
                "combined_label": row.get("combined_label", ""),
            }
        )

    start_date = min(dashboard_dates) if dashboard_dates else fallback_date
    end_date = max(dashboard_dates) if dashboard_dates else fallback_date
    return _build_scraping_dashboard(
        dashboard_rows,
        start_date,
        end_date,
        trend_subject="Data",
    )


def _apply_scraping_context(
    context: dict[str, object],
    rows: list[dict[str, object]],
    tweet_count: int,
    requested_page: int,
    per_page: int,
    start_date_value: object,
    end_date_value: object,
    history_id: int | None = None,
    dashboard_enabled: bool = False,
    query: str | None = None,
) -> bool:
    page_rows, total_rows, current_page, total_pages = _paginate_rows(rows, requested_page, per_page)
    return _apply_scraping_page_context(
        context,
        page_rows,
        total_rows,
        current_page,
        total_pages,
        tweet_count,
        per_page,
        start_date_value,
        end_date_value,
        history_id=history_id,
        dashboard_enabled=dashboard_enabled,
        query=query,
        rows=rows,
    )


def _apply_scraping_page_context(
    context: dict[str, object],
    page_rows: list[dict[str, object]],
    total_rows: int,
    current_page: int,
    total_pages: int,
    tweet_count: int,
    per_page: int,
    start_date_value: object,
    end_date_value: object,
    history_id: int | None = None,
    dashboard_enabled: bool = False,
    query: str | None = None,
    rows: list[dict[str, object]] | None = None,
) -> bool:
    if total_rows <= 0:
        return False

    page_start = max(1, current_page - 2)
    page_end = min(total_pages, current_page + 2)
    context["tweet_count"] = tweet_count or total_rows
    context["classified_preview"] = page_rows
    context["current_page"] = current_page
    context["total_pages"] = total_pages
    context["page_numbers"] = range(page_start, page_end + 1)
    context["per_page"] = per_page
    context["dashboard_enabled"] = bool(dashboard_enabled)
    if history_id:
        context["classified_download_url"] = reverse("download_scrape_history_csv", args=[history_id])
        context["classified_download_xlsx_url"] = reverse("download_scrape_history_xlsx", args=[history_id])
    if not dashboard_enabled:
        context["dashboard"] = None
        context["dashboard_error"] = ""
        return True

    try:
        context["dashboard"] = _build_scraping_dashboard(
            rows or [],
            _safe_parse_iso_date(start_date_value),
            _safe_parse_iso_date(end_date_value),
            query=query,
        )
        context["dashboard_error"] = ""
    except Exception:
        context["dashboard"] = None
        context["dashboard_error"] = (
            "Dashboard tidak dapat ditampilkan untuk hasil ini. "
            "Silakan gunakan rentang tanggal lebih pendek atau periksa data scraping."
        )
    return True


def _json_safe_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_value(val) for key, val in value.items()}

    # Numpy scalar / decimal-like values.
    if hasattr(value, "item"):
        try:
            return _json_safe_value(value.item())  # type: ignore[attr-defined]
        except Exception:
            pass

    return str(value)


def _safe_next_relative_url(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("//"):
        return ""
    if not text.startswith("/"):
        return ""
    return text


def _serialize_history_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    for row in rows:
        serialized.append({str(key): _json_safe_value(value) for key, value in row.items()})
    return serialized


def _safe_download_filename(base_name: str, fallback_prefix: str, extension: str = ".csv") -> str:
    stem = Path(str(base_name or "").strip()).stem
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    if not sanitized:
        sanitized = fallback_prefix
    safe_extension = extension if str(extension).startswith(".") else f".{extension}"
    return f"{sanitized}{safe_extension}"


def _export_cell_value(column: str, value: object) -> str:
    if value is None:
        return ""
    if column in PREDICTION_LABEL_COLUMNS:
        return _display_sentiment_label_id(value)
    if column in PREDICTION_SCORE_COLUMNS:
        try:
            return f"{float(value):.6f}"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _csv_download_response(
    filename: str,
    columns: list[str],
    headers: list[str],
    rows: list[dict[str, object]],
) -> HttpResponse:
    response = HttpResponse(content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    writer = csv.writer(response)
    writer.writerow(headers)
    for row in rows:
        writer.writerow([_export_cell_value(column, row.get(column, "")) for column in columns])
    return response


def _xlsx_inline_cell(value: str) -> str:
    escaped_value = escape(str(value or ""))
    return (
        '<c t="inlineStr">'
        f"<is><t>{escaped_value}</t></is>"
        "</c>"
    )


def _xlsx_download_response(
    filename: str,
    columns: list[str],
    headers: list[str],
    rows: list[dict[str, object]],
) -> HttpResponse:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>""",
        )
        archive.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>""",
        )
        archive.writestr(
            "docProps/core.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Export Hasil Klasifikasi</dc:title>
  <dc:creator>Sistem Analisis Sentimen</dc:creator>
</cp:coreProperties>""",
        )
        archive.writestr(
            "docProps/app.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>OpenAI Codex</Application>
</Properties>""",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>""",
        )
        archive.writestr(
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Data" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>""",
        )

        worksheet_rows: list[str] = []
        header_cells = "".join(_xlsx_inline_cell(header) for header in headers)
        worksheet_rows.append(f"<row r=\"1\">{header_cells}</row>")
        for index, row in enumerate(rows, start=2):
            cells = "".join(_xlsx_inline_cell(_export_cell_value(column, row.get(column, ""))) for column in columns)
            worksheet_rows.append(f"<row r=\"{index}\">{cells}</row>")
        sheet_xml = (
            """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>"""
            + "".join(worksheet_rows)
            + """</sheetData>
</worksheet>"""
        )
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)

    response = HttpResponse(
        output.getvalue(),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


def _combine_tweet_predictions(
    tweets: list[dict[str, object]],
    predictions: list[dict[str, object]],
) -> list[dict[str, object]]:
    combined_rows: list[dict[str, object]] = []
    for tweet, prediction in zip(tweets, predictions):
        combined_rows.append(
            {
                "id": tweet.get("id", ""),
                "url": tweet.get("url", ""),
                "text": tweet.get("text", prediction.get("text", "")),
                "retweetCount": tweet.get("retweetCount", ""),
                "replyCount": tweet.get("replyCount", ""),
                "likeCount": tweet.get("likeCount", ""),
                "quoteCount": tweet.get("quoteCount", ""),
                "viewCount": tweet.get("viewCount", ""),
                "CreatedAt": tweet.get("CreatedAt", ""),
                "lang": tweet.get("lang", ""),
                "bookmarkCount": tweet.get("bookmarkCount", ""),
                "isReply": tweet.get("isReply", ""),
                "inReplyTold": tweet.get("inReplyTold", ""),
                "userName": tweet.get("userName", ""),
                "image_tweet": tweet.get("image_tweet", ""),
                "_week_start": tweet.get("_week_start", ""),
                "_week_end": tweet.get("_week_end", ""),
                "knn_label": prediction.get("knn_label", ""),
                "knn_positive_score": prediction.get("knn_positive_score"),
                "knn_negative_score": prediction.get("knn_negative_score"),
                "svm_label": prediction.get("svm_label", ""),
                "svm_positive_score": prediction.get("svm_positive_score"),
                "svm_negative_score": prediction.get("svm_negative_score"),
                "combined_label": prediction.get("combined_label", ""),
                "combined_positive_score": prediction.get("combined_positive_score"),
                "combined_negative_score": prediction.get("combined_negative_score"),
            }
        )
    return combined_rows


def _filter_rows_by_date_range(
    rows: list[dict[str, object]],
    start_date: date | None,
    end_date: date | None,
) -> list[dict[str, object]]:
    filtered_rows: list[dict[str, object]] = []
    for row in rows:
        created_date = _parse_created_at_date(row.get("CreatedAt"))
        if created_date is None:
            created_date = _safe_parse_iso_date(row.get("_week_start"))
        if created_date is None:
            # Fallback: keep row to avoid dropping valid tweets due to format variation.
            filtered_rows.append(row)
            continue
        if start_date and created_date < start_date:
            continue
        if end_date and created_date > end_date:
            continue
        filtered_rows.append(row)
    return filtered_rows


def _load_scrape_rows(history: ScrapeHistory) -> list[dict[str, object]]:
    stored_rows = history.rows if isinstance(history.rows, list) else []
    rows: list[dict[str, object]] = list(stored_rows)
    for chunk_rows in history.temp_chunks.order_by("chunk_index", "id").values_list("rows", flat=True):
        if isinstance(chunk_rows, list):
            rows.extend(chunk_rows)
    return rows


def _tweet_dedup_key(row: dict[str, object]) -> str:
    tweet_id = str(row.get("id", "")).strip()
    if tweet_id:
        return f"id:{tweet_id}"

    dedup_user = str(row.get("userName", "")).strip().lower()
    dedup_created = str(row.get("CreatedAt", "")).strip()
    dedup_text = str(row.get("text", "")).strip().lower()
    return f"fallback:{dedup_user}|{dedup_created}|{dedup_text[:180]}"


def _build_existing_scrape_keys(history: ScrapeHistory) -> set[str]:
    return {_tweet_dedup_key(row) for row in _load_scrape_rows(history)}


def _next_chunk_index(history: ScrapeHistory) -> int:
    current_max = history.temp_chunks.aggregate(models.Max("chunk_index")).get("chunk_index__max")
    if current_max is None:
        return 0
    return int(current_max) + 1


def _append_rows_to_history_chunks(
    history: ScrapeHistory,
    rows: list[dict[str, object]],
    seen_keys: set[str],
    chunk_index: int,
) -> tuple[int, int]:
    deduped_rows: list[dict[str, object]] = []
    for row in rows:
        dedup_key = _tweet_dedup_key(row)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        deduped_rows.append(row)

    if not deduped_rows:
        return 0, chunk_index

    ScrapeTempChunk.objects.create(
        history=history,
        chunk_index=chunk_index,
        rows=_serialize_history_rows(deduped_rows),
    )
    return len(deduped_rows), chunk_index + 1


def _next_fetch_start_date(
    fetch_meta: dict[str, object],
    fallback_start_date: date,
) -> date:
    parsed = _safe_parse_iso_date(fetch_meta.get("next_start_date"))
    if parsed is None:
        return fallback_start_date
    return parsed


def _apply_history_progress_meta(
    history: ScrapeHistory,
    fetch_meta: dict[str, object],
    end_date: date,
    fallback_start_date: date,
) -> None:
    next_start_date = _next_fetch_start_date(fetch_meta, fallback_start_date)
    timed_out = bool(fetch_meta.get("timed_out"))
    rate_limited = bool(fetch_meta.get("rate_limited"))
    truncated = bool(fetch_meta.get("truncated"))
    is_finished = next_start_date > end_date and not (timed_out or rate_limited or truncated)

    history.is_complete = bool(is_finished)
    history.resume_next_date = None if history.is_complete else min(next_start_date, end_date)
    if history.is_complete:
        history.stop_reason = ""
    elif timed_out:
        history.stop_reason = "timed_out"
    elif rate_limited:
        history.stop_reason = "rate_limited"
    elif truncated:
        history.stop_reason = "truncated"
    else:
        history.stop_reason = "partial"


def _add_fetch_meta_messages(
    request: HttpRequest,
    fetch_meta: dict[str, object],
    current_count: int,
    effective_total_tweets: int,
    max_total_tweets: int,
) -> None:
    if bool(fetch_meta.get("rate_limited")):
        messages.warning(
            request,
            "Sebagian data berhasil diambil, tetapi proses berhenti karena batas permintaan API.",
        )
    if effective_total_tweets > 0 and (current_count >= effective_total_tweets or bool(fetch_meta.get("truncated"))):
        messages.warning(
            request,
            f"Hasil dibatasi maksimal {effective_total_tweets} tweet per scraping agar aplikasi tetap stabil.",
        )
    if effective_total_tweets > 0 and effective_total_tweets < max_total_tweets:
        messages.info(
            request,
            f"Untuk stabilitas server, rentang ini diproses dengan sampling maksimal {effective_total_tweets} tweet.",
        )


def _build_scrape_runtime_config(start_date: date, end_date: date) -> dict[str, int | bool]:
    max_total_tweets = _setting_positive_int("SENTIMENT_TWITTER_MAX_TOTAL_TWEETS", 4000)
    max_tweets_per_window = _setting_positive_int("SENTIMENT_TWITTER_MAX_TWEETS_PER_WINDOW", 500)
    min_tweets_per_window = _setting_positive_int("SENTIMENT_TWITTER_MIN_TWEETS_PER_WINDOW", 80)
    max_runtime_seconds = _setting_positive_int("SENTIMENT_TWITTER_MAX_RUNTIME_SECONDS", 90)
    predict_chunk_size = _setting_positive_int("SENTIMENT_TWITTER_PREDICT_CHUNK_SIZE", 300)
    temp_db_threshold_days = _setting_positive_int("SENTIMENT_TWITTER_TEMP_DB_THRESHOLD_DAYS", 90)

    selected_days = (end_date - start_date).days + 1
    use_temp_db_mode = selected_days > temp_db_threshold_days
    # Process daily to avoid historical-search drift where broader windows can
    # repeatedly return the same tweets and starve later dates.
    window_days = 1

    total_windows = max(1, (selected_days + window_days - 1) // window_days)
    effective_total_tweets = max_total_tweets
    effective_tweets_per_window = min(
        max_tweets_per_window,
        max(min_tweets_per_window, (effective_total_tweets + total_windows - 1) // total_windows),
    )

    return {
        "selected_days": selected_days,
        "use_temp_db_mode": use_temp_db_mode,
        "window_days": window_days,
        "effective_total_tweets": effective_total_tweets,
        "effective_tweets_per_window": effective_tweets_per_window,
        "max_total_tweets": max_total_tweets,
        "max_runtime_seconds": max_runtime_seconds,
        "predict_chunk_size": predict_chunk_size,
    }


def _finish_prediction_history_with_error(history_id: int, message: str) -> None:
    PredictionHistory.objects.filter(id=history_id).update(
        is_processing=False,
        error_message=str(message or "").strip(),
        sample_count=0,
        columns=[],
        rows=[],
    )


def _process_prediction_history_file_job(
    history_id: int,
    uploaded_name: str,
    uploaded_content: bytes,
    uploaded_content_type: str,
    selected_text_column: str,
    model_version: str,
) -> None:
    try:
        resolved_model_version = resolve_model_version_name(model_version)
        uploaded_file = SimpleUploadedFile(
            uploaded_name,
            uploaded_content,
            content_type=uploaded_content_type or "application/octet-stream",
        )
        texts, detected_column, source_rows, source_columns = parse_uploaded_file(
            uploaded_file,
            selected_text_column or None,
        )
        predictions = predict_batch(texts, model_version=resolved_model_version or None)
        merged_rows = _merge_batch_rows_for_history(source_rows, predictions)
        PredictionHistory.objects.filter(id=history_id).update(
            model_version=resolved_model_version,
            text_column=(detected_column or selected_text_column or "").strip(),
            sample_count=len(merged_rows),
            columns=[str(column) for column in source_columns],
            rows=_serialize_history_rows(merged_rows),
            is_processing=False,
            error_message="",
        )
    except (ModelServiceError, FileValidationError) as exc:
        _finish_prediction_history_with_error(history_id, str(exc))
    except Exception as exc:
        _finish_prediction_history_with_error(
            history_id,
            f"Terjadi kesalahan tak terduga saat menjalankan prediksi: {exc}",
        )


def _finish_scrape_history_with_error(history_id: int, message: str) -> None:
    ScrapeHistory.objects.filter(id=history_id).update(
        is_processing=False,
        is_complete=False,
        stop_reason="error",
        error_message=str(message or "").strip(),
    )


def _process_scrape_history_job(history_id: int, api_key: str) -> None:
    retry_attempts = 0

    while True:
        history = ScrapeHistory.objects.filter(id=history_id).first()
        if history is None:
            return

        if not history.is_processing:
            return

        try:
            resume_result = _resume_scrape_once(history, api_key)
        except Exception as exc:
            _finish_scrape_history_with_error(
                history_id,
                f"Terjadi kesalahan tak terduga saat mengambil/mengklasifikasikan tweet: {exc}",
            )
            return

        if not bool(resume_result.get("ok")):
            if bool(resume_result.get("retryable")) and retry_attempts < BACKGROUND_HISTORY_RETRY_LIMIT:
                retry_attempts += 1
                retry_after_seconds = int(
                    resume_result.get("retry_after_seconds") or BACKGROUND_HISTORY_RETRY_DELAY_SECONDS
                )
                time.sleep(max(1, retry_after_seconds))
                continue

            _finish_scrape_history_with_error(
                history_id,
                str(resume_result.get("error") or "Terjadi kesalahan saat memproses scraping."),
            )
            return

        retry_attempts = 0
        history.refresh_from_db()

        if bool(resume_result.get("complete")):
            error_message = ""
            stop_reason = history.stop_reason
            if int(history.tweet_count or 0) <= 0:
                error_message = "Tidak ada tweet yang ditemukan untuk permintaan ini."
                stop_reason = "empty"

            ScrapeHistory.objects.filter(id=history_id).update(
                is_processing=False,
                stop_reason=stop_reason,
                error_message=error_message,
            )
            return

        time.sleep(0.2)


def _history_resume_progress(history: ScrapeHistory) -> tuple[int, int, int]:
    total_days = max(1, (history.end_date - history.start_date).days + 1)
    if history.is_complete:
        done_days = total_days
    else:
        next_date = history.resume_next_date or history.start_date
        done_days = max(0, min(total_days, (next_date - history.start_date).days))
    progress_pct = int((done_days * 100) / total_days)
    return done_days, total_days, progress_pct


def _resume_scrape_once(history: ScrapeHistory, api_key: str) -> dict[str, object]:
    resolved_model_version = str(history.model_version or "").strip()
    if not resolved_model_version:
        resolved_model_version = resolve_model_version_name(None)
        if resolved_model_version:
            history.model_version = resolved_model_version
            ScrapeHistory.objects.filter(id=history.id, model_version="").update(model_version=resolved_model_version)

    if history.is_complete:
        done_days, total_days, progress_pct = _history_resume_progress(history)
        return {
            "ok": True,
            "complete": True,
            "appended_count": 0,
            "tweet_count": int(history.tweet_count or 0),
            "resume_next_date": "",
            "stop_reason": "",
            "rate_limited": False,
            "timed_out": False,
            "truncated": False,
            "progress_pct": progress_pct,
            "done_days": done_days,
            "total_days": total_days,
            "effective_total_tweets": 0,
            "max_total_tweets": 0,
        }

    resume_start_date = history.resume_next_date or history.start_date
    if resume_start_date > history.end_date:
        history.is_complete = True
        history.resume_next_date = None
        history.stop_reason = ""
        history.save(update_fields=["is_complete", "resume_next_date", "stop_reason"])
        done_days, total_days, progress_pct = _history_resume_progress(history)
        return {
            "ok": True,
            "complete": True,
            "appended_count": 0,
            "tweet_count": int(history.tweet_count or 0),
            "resume_next_date": "",
            "stop_reason": "",
            "rate_limited": False,
            "timed_out": False,
            "truncated": False,
            "progress_pct": progress_pct,
            "done_days": done_days,
            "total_days": total_days,
            "effective_total_tweets": 0,
            "max_total_tweets": 0,
        }

    resume_window_end_date = min(history.end_date, resume_start_date + timedelta(days=1) - timedelta(days=1))
    runtime_config = _build_scrape_runtime_config(resume_start_date, history.end_date)
    effective_total_tweets = int(runtime_config["effective_total_tweets"])
    effective_tweets_per_window = int(runtime_config["effective_tweets_per_window"])
    max_total_tweets = int(runtime_config["max_total_tweets"])
    max_runtime_seconds = int(runtime_config["max_runtime_seconds"])
    predict_chunk_size = int(runtime_config["predict_chunk_size"])
    window_days = int(runtime_config["window_days"])

    seen_keys = _build_existing_scrape_keys(history)
    chunk_index = _next_chunk_index(history)
    appended_count = 0
    current_tweet_count = int(history.tweet_count or 0)
    remaining_total_capacity = max(1, max_total_tweets - current_tweet_count)

    def _handle_window(window_tweets: list[dict[str, object]]) -> None:
        nonlocal chunk_index, appended_count, current_tweet_count
        if not window_tweets:
            return

        predictions = predict_batch_in_chunks(
            [str(tweet.get("text", "")) for tweet in window_tweets],
            chunk_size=predict_chunk_size,
            model_version=resolved_model_version or None,
        )
        classified_rows = _combine_tweet_predictions(window_tweets, predictions)
        filtered_rows = _filter_rows_by_date_range(classified_rows, history.start_date, history.end_date)
        if not filtered_rows:
            return

        new_count, next_chunk_index = _append_rows_to_history_chunks(
            history,
            filtered_rows,
            seen_keys,
            chunk_index,
        )
        chunk_index = next_chunk_index
        appended_count += new_count
        current_tweet_count += new_count

    def _checkpoint_window(window_meta: dict[str, object]) -> None:
        next_start_text = str(window_meta.get("next_start_date") or "").strip()
        next_start_date = _safe_parse_iso_date(next_start_text)
        if next_start_date is None:
            return

        ScrapeHistory.objects.filter(id=history.id).update(
            tweet_count=current_tweet_count,
            resume_next_date=min(next_start_date, history.end_date),
            is_complete=False,
            stop_reason="processing",
        )

    try:
        fetch_result = fetch_tweets(
            api_key=api_key,
            query=history.query,
            language=history.language,
            start_date=resume_start_date.isoformat(),
            end_date=resume_window_end_date.isoformat(),
            max_tweets_per_window=effective_tweets_per_window,
            max_total_tweets=min(effective_total_tweets, remaining_total_capacity),
            window_days=window_days,
            max_runtime_seconds=max_runtime_seconds,
            on_window=_handle_window,
            on_window_checkpoint=_checkpoint_window,
            return_meta=True,
            existing_tweet_keys=set(seen_keys),
        )
    except TwitterRateLimitError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "retryable": True,
            "retry_after_seconds": 8,
            "error_code": "rate_limited",
        }
    except TwitterTimeoutError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "retryable": True,
            "retry_after_seconds": 3,
            "error_code": "timed_out",
        }
    except (TwitterAPIError, ModelServiceError, FileValidationError) as exc:
        return {"ok": False, "error": str(exc), "retryable": False}
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Terjadi kesalahan tak terduga saat melanjutkan scraping: {exc}",
            "retryable": False,
        }

    fetch_meta: dict[str, object] = {
        "rate_limited": False,
        "truncated": False,
        "timed_out": False,
        "next_start_date": (history.end_date + timedelta(days=1)).isoformat(),
    }
    if isinstance(fetch_result, tuple) and len(fetch_result) == 2 and isinstance(fetch_result[1], dict):
        fetch_meta = fetch_result[1]

    total_rows = _load_scrape_rows(history)
    history.tweet_count = len(total_rows)
    _apply_history_progress_meta(history, fetch_meta, history.end_date, resume_start_date)
    history.save(update_fields=["tweet_count", "is_complete", "resume_next_date", "stop_reason"])
    done_days, total_days, progress_pct = _history_resume_progress(history)

    return {
        "ok": True,
        "complete": bool(history.is_complete),
        "appended_count": int(appended_count),
        "tweet_count": int(history.tweet_count or 0),
        "resume_next_date": history.resume_next_date.isoformat() if history.resume_next_date else "",
        "stop_reason": str(history.stop_reason or ""),
        "rate_limited": bool(fetch_meta.get("rate_limited")),
        "timed_out": bool(fetch_meta.get("timed_out")),
        "truncated": bool(fetch_meta.get("truncated")),
        "progress_pct": progress_pct,
        "done_days": done_days,
        "total_days": total_days,
        "effective_total_tweets": effective_total_tweets,
        "max_total_tweets": max_total_tweets,
    }


def login_view(request: HttpRequest) -> HttpResponse:
    if request.user.is_authenticated:
        return redirect("home")

    form = LoginForm(request, data=request.POST or None)
    if request.method == "POST" and form.is_valid():
        auth_login(request, form.get_user())
        messages.success(request, "Login berhasil.")
        return redirect("home")
    if request.method == "POST":
        messages.error(request, "Username atau password tidak valid.")

    return render(request, "sentiment_app/login.html", {"form": form})


@login_required
def logout_view(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        auth_logout(request)
        messages.info(request, "Anda sudah logout.")
        return redirect("login")
    return redirect("home")


def _require_superuser(request: HttpRequest) -> None:
    if not request.user.is_superuser:
        raise PermissionDenied("Hanya Administrator yang dapat mengakses Admin Panel.")


def _scrape_history_queryset_for_detail(request: HttpRequest):
    if request.user.is_superuser:
        return ScrapeHistory.objects.all()
    return ScrapeHistory.objects.filter(user=request.user)


def _prediction_history_queryset_for_detail(request: HttpRequest):
    if request.user.is_superuser:
        return PredictionHistory.objects.all()
    return PredictionHistory.objects.filter(user=request.user)


def _prediction_dataset_queryset():
    return PredictionHistory.objects.all()


def _user_prediction_dataset_queryset(user):
    return _prediction_dataset_queryset().filter(user=user)


def _delete_file_field(file_field) -> None:
    if not file_field:
        return
    storage = file_field.storage
    file_name = str(file_field.name or "").strip()
    if file_name:
        storage.delete(file_name)


def _save_storage_file_overwrite(storage, target_name: str, content_file) -> str:
    if storage.exists(target_name):
        storage.delete(target_name)

    saved_name = storage.save(target_name, content_file)
    if saved_name != target_name:
        if storage.exists(saved_name):
            storage.delete(saved_name)
        raise RuntimeError(f"Gagal menyimpan file model ke nama stabil: {target_name}")

    return saved_name


def _save_uploaded_model_version(version_name: str, knn_file, svm_file) -> SentimentModelVersion:
    record = SentimentModelVersion(version_name=version_name)
    record.save()
    _update_model_version_files(record, knn_file=knn_file, svm_file=svm_file)
    clear_cache()
    return record


def _save_model_file_to_stable_path(
    record: SentimentModelVersion,
    field_name: str,
    uploaded_file,
    kind: str,
) -> str:
    if uploaded_file is None:
        return str(getattr(record, field_name).name or "").strip()

    field_file = getattr(record, field_name)
    storage = field_file.storage
    old_name = str(field_file.name or "").strip()
    target_name = sentiment_model_storage_name(record.version_name, kind, getattr(uploaded_file, "name", ""))

    if old_name and old_name != target_name:
        storage.delete(old_name)

    if hasattr(uploaded_file, "seek"):
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

    saved_name = _save_storage_file_overwrite(storage, target_name, uploaded_file)
    setattr(record, field_name, saved_name)
    return saved_name


def _model_version_directory(version_name: str) -> SentimentModelVersion:
    normalized_name = str(version_name or "").strip()
    if not normalized_name:
        raise Http404("Versi model tidak ditemukan.")
    record = SentimentModelVersion.objects.filter(version_name=normalized_name).first()
    if record is None:
        raise Http404("Versi model tidak ditemukan.")
    return record


def _update_model_version_files(record: SentimentModelVersion, knn_file=None, svm_file=None) -> None:
    updated_fields: list[str] = []
    if knn_file is not None:
        _save_model_file_to_stable_path(record, "knn_model_file", knn_file, "knn")
        updated_fields.append("knn_model_file")
    if svm_file is not None:
        _save_model_file_to_stable_path(record, "svm_model_file", svm_file, "svm")
        updated_fields.append("svm_model_file")
    if updated_fields:
        updated_fields.append("updated_at")
        record.save(update_fields=updated_fields)


def _move_model_file_to_version_path(record: SentimentModelVersion, field_name: str, kind: str) -> None:
    field_file = getattr(record, field_name)
    storage = field_file.storage
    old_name = str(field_file.name or "").strip()
    if not old_name:
        return

    target_name = sentiment_model_storage_name(record.version_name, kind, old_name)
    if old_name == target_name:
        return

    with storage.open(old_name, "rb") as source_file:
        saved_name = _save_storage_file_overwrite(storage, target_name, File(source_file, name=Path(target_name).name))

    storage.delete(old_name)
    setattr(record, field_name, saved_name)


def _rename_model_version(old_version_name: str, new_version_name: str) -> SentimentModelVersion:
    record = _model_version_directory(old_version_name)
    if old_version_name == new_version_name:
        return record
    record.version_name = new_version_name
    _move_model_file_to_version_path(record, "knn_model_file", "knn")
    _move_model_file_to_version_path(record, "svm_model_file", "svm")
    record.save(update_fields=["version_name", "knn_model_file", "svm_model_file", "updated_at"])
    ScrapeHistory.objects.filter(model_version=old_version_name).update(model_version=new_version_name)
    PredictionHistory.objects.filter(model_version=old_version_name).update(model_version=new_version_name)
    return record


@login_required
def admin_dashboard_view(request: HttpRequest) -> HttpResponse:
    _require_superuser(request)

    users_qs = User.objects.order_by("username").only(
        "id",
        "username",
        "first_name",
        "last_name",
        "email",
        "is_active",
        "is_staff",
        "is_superuser",
        "date_joined",
        "last_login",
    )
    scrape_histories_qs = ScrapeHistory.objects.select_related("user").only(
        "id",
        "user__username",
        "query",
        "language",
        "start_date",
        "end_date",
        "tweet_count",
        "is_complete",
        "created_at",
    )
    prediction_histories_qs = _prediction_dataset_queryset().select_related("user").only(
        "id",
        "user__username",
        "source_name",
        "text_column",
        "sample_count",
        "created_at",
    )

    users_page_obj = Paginator(users_qs, 3).get_page(
        _safe_positive_int(request.GET.get("users_page"), 1)
    )
    scrape_page_obj = Paginator(scrape_histories_qs, HISTORY_PER_PAGE).get_page(
        _safe_positive_int(request.GET.get("scrape_page"), 1)
    )
    prediction_page_obj = Paginator(prediction_histories_qs, HISTORY_PER_PAGE).get_page(
        _safe_positive_int(request.GET.get("pred_page"), 1)
    )
    model_versions = list(SentimentModelVersion.objects.order_by("version_name", "id"))

    context = {
        "users_page_obj": users_page_obj,
        "users": users_page_obj.object_list,
        "scrape_page_obj": scrape_page_obj,
        "scrape_histories": scrape_page_obj.object_list,
        "prediction_page_obj": prediction_page_obj,
        "prediction_histories": prediction_page_obj.object_list,
        "total_users": User.objects.count(),
        "total_staff": User.objects.filter(is_staff=True, is_superuser=False).count(),
        "total_superusers": User.objects.filter(is_superuser=True).count(),
        "total_scrape_histories": ScrapeHistory.objects.count(),
        "total_prediction_histories": _prediction_dataset_queryset().count(),
        "total_model_versions": len(model_versions),
        "model_versions": model_versions,
    }
    return render(request, "sentiment_app/admin_panel.html", context)


@login_required
def admin_model_create_view(request: HttpRequest) -> HttpResponse:
    _require_superuser(request)
    form = AdminModelUploadForm(request.POST or None, request.FILES or None)
    if request.method == "POST" and form.is_valid():
        version_name = form.cleaned_data["version_name"]
        _save_uploaded_model_version(
            version_name,
            form.cleaned_data["knn_model_file"],
            form.cleaned_data["svm_model_file"],
        )
        messages.success(request, f"Versi model {version_name} berhasil ditambahkan.")
        return redirect("admin:index")

    context = {
        "form": form,
        "form_title": "Tambah Model",
        "form_description": "Unggah satu versi model baru dengan file KNN dan SVM.",
        "submit_label": "Simpan Model",
    }
    return render(request, "sentiment_app/admin_model_form.html", context)


@login_required
def admin_model_edit_view(request: HttpRequest, version_name: str) -> HttpResponse:
    _require_superuser(request)
    target_record = _model_version_directory(version_name)
    form = AdminModelEditForm(
        request.POST or None,
        request.FILES or None,
        existing_version_name=version_name,
        initial={"version_name": version_name},
    )
    if request.method == "POST" and form.is_valid():
        new_version_name = form.cleaned_data["version_name"]
        updated_dir = _rename_model_version(version_name, new_version_name)
        _update_model_version_files(
            updated_dir,
            form.cleaned_data.get("knn_model_file"),
            form.cleaned_data.get("svm_model_file"),
        )
        clear_cache()
        messages.success(request, f"Versi model {new_version_name} berhasil diperbarui.")
        return redirect("admin:index")

    context = {
        "form": form,
        "form_title": "Edit Model",
        "form_description": "Ubah nama versi model atau ganti file KNN dan SVM.",
        "submit_label": "Simpan Perubahan",
        "target_model_version": target_record,
    }
    return render(request, "sentiment_app/admin_model_form.html", context)


@login_required
@require_POST
def admin_model_delete_view(request: HttpRequest, version_name: str) -> HttpResponse:
    _require_superuser(request)
    target_record = _model_version_directory(version_name)
    knn_file = target_record.knn_model_file
    svm_file = target_record.svm_model_file
    target_record.delete()
    _delete_file_field(knn_file)
    _delete_file_field(svm_file)
    clear_cache()
    messages.success(request, f"Versi model {version_name} berhasil dihapus.")
    return redirect("admin:index")


@login_required
def admin_user_create_view(request: HttpRequest) -> HttpResponse:
    _require_superuser(request)
    form = AdminCreateUserForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user_obj = form.save()
        messages.success(request, f"User {user_obj.username} berhasil dibuat.")
        return redirect("admin:index")

    context = {
        "form": form,
        "form_title": "Tambah User",
        "form_description": "Tambah data akun",
        "submit_label": "Simpan User",
        "target_user": None,
    }
    return render(request, "sentiment_app/admin_user_form.html", context)


@login_required
def admin_user_edit_view(request: HttpRequest, user_id: int) -> HttpResponse:
    _require_superuser(request)
    target_user = get_object_or_404(User, pk=user_id)
    form = AdminEditUserForm(request.POST or None, instance=target_user)
    if request.method == "POST" and form.is_valid():
        keeps_current_admin = (
            target_user.pk != request.user.pk
            or (
                bool(form.cleaned_data.get("is_active"))
                and form.cleaned_data.get("role") == "administrator"
            )
        )
        if not keeps_current_admin:
            form.add_error(None, "Tidak bisa mencabut akses admin dari akun yang sedang login.")
        else:
            user_obj = form.save()
            messages.success(request, f"User {user_obj.username} berhasil diperbarui.")
            return redirect("admin:index")

    context = {
        "form": form,
        "form_title": "Edit User",
        "form_description": "Edit data akun",
        "submit_label": "Simpan Perubahan",
        "target_user": target_user,
    }
    return render(request, "sentiment_app/admin_user_form.html", context)


@login_required
@require_POST
def admin_user_delete_view(request: HttpRequest, user_id: int) -> HttpResponse:
    _require_superuser(request)
    target_user = get_object_or_404(User, pk=user_id)
    if target_user.pk == request.user.pk:
        messages.error(request, "User yang sedang login tidak bisa dihapus.")
        return redirect("admin:index")

    username = target_user.username
    target_user.delete()
    messages.success(request, f"User {username} berhasil dihapus.")
    return redirect("admin:index")


@login_required
def admin_prediction_history_edit_view(request: HttpRequest, history_id: int) -> HttpResponse:
    _require_superuser(request)
    history = get_object_or_404(PredictionHistory, pk=history_id)
    form = AdminPredictionHistoryForm(request.POST or None, instance=history)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "PredictionHistory berhasil diperbarui.")
        return redirect("admin:index")

    context = {
        "form": form,
        "form_title": "Edit PredictionHistory",
        "submit_label": "Simpan PredictionHistory",
        "detail_url": reverse("prediction_history_detail", args=[history.id]),
        "delete_url": reverse("admin:prediction_history_delete", args=[history.id]),
        "delete_label": "Hapus PredictionHistory",
    }
    return render(request, "sentiment_app/admin_history_form.html", context)


@login_required
@require_POST
def admin_prediction_history_delete_view(request: HttpRequest, history_id: int) -> HttpResponse:
    _require_superuser(request)
    history = get_object_or_404(PredictionHistory, pk=history_id)
    history.delete()
    messages.success(request, "PredictionHistory berhasil dihapus.")
    return redirect("admin:index")


@login_required
def admin_scrape_history_edit_view(request: HttpRequest, history_id: int) -> HttpResponse:
    _require_superuser(request)
    history = get_object_or_404(ScrapeHistory, pk=history_id)
    form = AdminScrapeHistoryForm(request.POST or None, instance=history)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "ScrapeHistory berhasil diperbarui.")
        return redirect("admin:index")

    context = {
        "form": form,
        "form_title": "Edit ScrapeHistory",
        "submit_label": "Simpan ScrapeHistory",
        "detail_url": reverse("history_detail", args=[history.id]),
        "delete_url": reverse("admin:scrape_history_delete", args=[history.id]),
        "delete_label": "Hapus ScrapeHistory",
    }
    return render(request, "sentiment_app/admin_history_form.html", context)


@login_required
@require_POST
def admin_scrape_history_delete_view(request: HttpRequest, history_id: int) -> HttpResponse:
    _require_superuser(request)
    history = get_object_or_404(ScrapeHistory, pk=history_id)
    history.delete()
    messages.success(request, "ScrapeHistory berhasil dihapus.")
    return redirect("admin:index")


def _admin_dashboard_redirect(request: HttpRequest) -> HttpResponse:
    users_page = _safe_positive_int(request.POST.get("users_page"), 1)
    scrape_page = _safe_positive_int(request.POST.get("scrape_page"), 1)
    pred_page = _safe_positive_int(request.POST.get("pred_page"), 1)
    dataset_tab = str(request.POST.get("dataset_tab") or "").strip().lower()
    if dataset_tab not in {"scraping", "prediction"}:
        dataset_tab = "prediction"
    return redirect(
        f"{reverse('admin:index')}?users_page={users_page}&scrape_page={scrape_page}&pred_page={pred_page}&dataset_tab={dataset_tab}"
    )


@login_required
@require_POST
def admin_delete_selected_history_view(request: HttpRequest) -> HttpResponse:
    _require_superuser(request)
    scope = str(request.POST.get("scope", "")).strip().lower()
    selected_ids_raw = request.POST.getlist("selected_ids")

    selected_ids: set[int] = set()
    for raw_id in selected_ids_raw:
        parsed_id = _safe_positive_int(raw_id, 0)
        if parsed_id > 0:
            selected_ids.add(parsed_id)

    if not selected_ids:
        messages.warning(request, "Pilih minimal satu dataset yang ingin dihapus.")
        return _admin_dashboard_redirect(request)

    if scope == "scrape":
        queryset = ScrapeHistory.objects.filter(id__in=selected_ids)
        deleted_histories = queryset.count()
        queryset.delete()
        messages.success(request, f"Dataset scraping terpilih berhasil dihapus ({deleted_histories} data).")
    elif scope == "prediction":
        queryset = _prediction_dataset_queryset().filter(id__in=selected_ids)
        deleted_histories = queryset.count()
        queryset.delete()
        messages.success(request, f"Dataset prediksi terpilih berhasil dihapus ({deleted_histories} data).")
    else:
        messages.warning(request, "Jenis dataset tidak valid.")

    return _admin_dashboard_redirect(request)


@login_required
def history_list_view(request: HttpRequest) -> HttpResponse:
    scrape_histories_qs = ScrapeHistory.objects.filter(user=request.user).only(
        "id",
        "query",
        "language",
        "start_date",
        "end_date",
        "tweet_count",
        "is_complete",
        "created_at",
    )
    prediction_histories_qs = _user_prediction_dataset_queryset(request.user).only(
        "id",
        "source_name",
        "text_column",
        "sample_count",
        "created_at",
    )

    scrape_page_number = _safe_positive_int(request.GET.get("scrape_page"), 1)
    prediction_page_number = _safe_positive_int(request.GET.get("pred_page"), 1)
    active_history_tab = str(request.GET.get("history_tab") or "").strip().lower()
    if active_history_tab not in {"scraping", "prediction"}:
        active_history_tab = "scraping"

    scrape_page_obj = Paginator(scrape_histories_qs, HISTORY_PER_PAGE).get_page(scrape_page_number)
    prediction_page_obj = Paginator(prediction_histories_qs, HISTORY_PER_PAGE).get_page(prediction_page_number)

    context = {
        "scrape_histories": scrape_page_obj.object_list,
        "prediction_histories": prediction_page_obj.object_list,
        "scrape_page_obj": scrape_page_obj,
        "prediction_page_obj": prediction_page_obj,
        "active_history_tab": active_history_tab,
    }
    return render(request, "sentiment_app/history.html", context)


def _history_list_redirect(request: HttpRequest) -> HttpResponse:
    scrape_page = _safe_positive_int(request.POST.get("scrape_page"), 1)
    pred_page = _safe_positive_int(request.POST.get("pred_page"), 1)
    history_tab = str(request.POST.get("history_tab") or "").strip().lower()
    if history_tab not in {"scraping", "prediction"}:
        history_tab = "scraping"
    return redirect(
        f"{reverse('history_list')}?scrape_page={scrape_page}&pred_page={pred_page}&history_tab={history_tab}"
    )


@login_required
@require_POST
def delete_scrape_history_view(request: HttpRequest, history_id: int) -> HttpResponse:
    history = get_object_or_404(ScrapeHistory, id=history_id, user=request.user)
    history.delete()
    messages.success(request, "Riwayat scraping berhasil dihapus.")
    return _history_list_redirect(request)


@login_required
@require_POST
def delete_prediction_history_view(request: HttpRequest, history_id: int) -> HttpResponse:
    history = get_object_or_404(PredictionHistory, id=history_id, user=request.user)
    history.delete()
    messages.success(request, "Riwayat prediksi berhasil dihapus.")
    return _history_list_redirect(request)


@login_required
@require_POST
def delete_all_history_view(request: HttpRequest) -> HttpResponse:
    scope = str(request.POST.get("scope", "")).strip().lower()
    if scope == "scrape":
        queryset = ScrapeHistory.objects.filter(user=request.user)
        deleted_histories = queryset.count()
        queryset.delete()
        messages.success(request, f"Semua riwayat scraping dihapus ({deleted_histories} data).")
    elif scope == "prediction":
        queryset = _user_prediction_dataset_queryset(request.user)
        deleted_histories = queryset.count()
        queryset.delete()
        messages.success(request, f"Semua riwayat prediksi dihapus ({deleted_histories} data).")
    else:
        messages.warning(request, "Pilih jenis riwayat yang ingin dihapus (scraping atau prediksi).")
    return _history_list_redirect(request)


@login_required
@require_POST
def delete_selected_history_view(request: HttpRequest) -> HttpResponse:
    scope = str(request.POST.get("scope", "")).strip().lower()
    selected_ids_raw = request.POST.getlist("selected_ids")

    selected_ids: set[int] = set()
    for raw_id in selected_ids_raw:
        parsed_id = _safe_positive_int(raw_id, 0)
        if parsed_id > 0:
            selected_ids.add(parsed_id)

    if not selected_ids:
        messages.warning(request, "Pilih minimal satu riwayat yang ingin dihapus.")
        return _history_list_redirect(request)

    if scope == "scrape":
        queryset = ScrapeHistory.objects.filter(user=request.user, id__in=selected_ids)
        deleted_histories = queryset.count()
        queryset.delete()
        messages.success(request, f"Riwayat scraping terpilih berhasil dihapus ({deleted_histories} data).")
    elif scope == "prediction":
        queryset = PredictionHistory.objects.filter(user=request.user, id__in=selected_ids)
        deleted_histories = queryset.count()
        queryset.delete()
        messages.success(request, f"Riwayat prediksi terpilih berhasil dihapus ({deleted_histories} data).")
    else:
        messages.warning(request, "Jenis riwayat tidak valid.")

    return _history_list_redirect(request)


@login_required
def history_detail_view(request: HttpRequest, history_id: int) -> HttpResponse:
    history = get_object_or_404(_scrape_history_queryset_for_detail(request), id=history_id)
    context = _build_scrape_history_view_context(request, history, show_dashboard=False)
    return render(request, "sentiment_app/twitter.html", context)


@login_required
def history_dashboard_view(request: HttpRequest, history_id: int) -> HttpResponse:
    history = get_object_or_404(_scrape_history_queryset_for_detail(request), id=history_id)
    context = _build_scrape_history_view_context(request, history, show_dashboard=True)
    return render(request, "sentiment_app/twitter.html", context)


@login_required
def history_dashboard_content_view(request: HttpRequest, history_id: int) -> HttpResponse:
    history = get_object_or_404(_scrape_history_queryset_for_detail(request), id=history_id)
    history = _upgrade_scrape_history_scores_if_needed(history)
    rows = _load_scrape_rows(history)

    dashboard = None
    dashboard_error = ""
    if rows:
        try:
            dashboard = _build_scraping_dashboard(
                rows,
                history.start_date,
                history.end_date,
                query=history.query,
            )
        except Exception:
            dashboard_error = (
                "Dashboard tidak dapat ditampilkan untuk hasil ini. "
                "Silakan gunakan rentang tanggal lebih pendek atau periksa data scraping."
            )
    else:
        dashboard_error = "Data riwayat scraping kosong."

    html = render_to_string(
        "sentiment_app/_history_scrape_dashboard_panel.html",
        {
            "dashboard": dashboard,
            "dashboard_error": dashboard_error,
        },
        request=request,
    )
    return JsonResponse({"html": html})


def _build_scrape_history_view_context(
    request: HttpRequest,
    history: ScrapeHistory,
    show_dashboard: bool,
) -> dict[str, object]:
    form = TwitterFetchForm()
    history_processing = bool(history.is_processing)
    history_error = str(history.error_message or "").strip()
    show_resume_form = not history_processing and not history_error and not history.is_complete
    resume_form = ResumeScrapeForm() if show_resume_form else None
    resume_done_days, resume_total_days, resume_progress_pct = _history_resume_progress(history)
    requested_page = _safe_positive_int(request.GET.get("page"), 1)
    per_page = _normalize_per_page(request.GET.get("per_page"), DEFAULT_PER_PAGE)
    dashboard_enabled = bool(show_dashboard)
    history_table_url = reverse("history_detail", args=[history.id])
    history_dashboard_url = reverse("history_dashboard", args=[history.id])

    context: dict[str, object] = {
        "form": form,
        "resume_form": resume_form,
        "history_mode": True,
        "history": history,
        "history_processing": history_processing,
        "history_error": history_error,
        "history_status_visible": bool(history_processing or history_error or not history.is_complete),
        "resume_done_days": resume_done_days,
        "resume_total_days": resume_total_days,
        "resume_progress_pct": resume_progress_pct,
        "resume_next_url": request.get_full_path(),
        "auto_resume_default": str(request.GET.get("auto", "")).strip() == "1",
        "dashboard_enabled": dashboard_enabled,
        "show_history_table": not dashboard_enabled,
        "show_history_dashboard": dashboard_enabled,
        "history_table_url": history_table_url,
        "history_dashboard_url": history_dashboard_url,
        "history_dashboard_content_url": reverse("history_dashboard_content", args=[history.id]),
        "history_title": "Dashboard Riwayat Pengumpulan Data X" if dashboard_enabled else "Detail Riwayat Pengumpulan Data X",
    }
    if dashboard_enabled or history_processing or bool(history_error):
        return context

    context["history_combined_counts"] = _build_scrape_history_combined_counts(history)

    history = _upgrade_scrape_history_scores_if_needed(history)
    has_context = False
    page_rows, total_rows, current_page, total_pages = _paginate_scrape_history_rows(history, requested_page, per_page)
    has_context = _apply_scraping_page_context(
        context,
        page_rows,
        total_rows,
        current_page,
        total_pages,
        int(history.tweet_count or 0),
        per_page,
        history.start_date.isoformat(),
        history.end_date.isoformat(),
        history_id=history.id,
        dashboard_enabled=False,
    )

    if not has_context and history.is_complete:
        messages.warning(request, "Data riwayat scraping kosong.")

    return context


@login_required
def resume_scrape_view(request: HttpRequest, history_id: int) -> HttpResponse:
    if request.method != "POST":
        return redirect("history_detail", history_id=history_id)

    wants_json = (
        request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or str(request.POST.get("ajax", "")).strip() == "1"
    )

    history = get_object_or_404(ScrapeHistory, id=history_id, user=request.user)
    next_url = _safe_next_relative_url(request.POST.get("next_url"))
    fallback_url = reverse("history_detail", args=[history.id])
    redirect_target = next_url or fallback_url
    form = ResumeScrapeForm(request.POST)
    if not form.is_valid():
        if wants_json:
            first_error = "Form tidak valid."
            for _field_name, field_errors in form.errors.items():
                if field_errors:
                    first_error = str(field_errors[0])
                    break
            return JsonResponse({"ok": False, "error": first_error}, status=400)
        for field_name, field_errors in form.errors.items():
            label = form.fields.get(field_name).label if field_name in form.fields else field_name
            for field_error in field_errors:
                messages.error(request, f"{label}: {field_error}")
        return redirect(redirect_target)

    api_key = (form.cleaned_data.get("api_key") or "").strip()
    if not api_key:
        if wants_json:
            return JsonResponse({"ok": False, "error": "API key wajib diisi."}, status=400)
        messages.error(request, "API key wajib diisi.")
        return redirect(redirect_target)

    resume_result = _resume_scrape_once(history, api_key)
    if not bool(resume_result.get("ok")):
        error_message = str(resume_result.get("error") or "Terjadi kesalahan saat melanjutkan scraping.")
        if wants_json:
            retryable = bool(resume_result.get("retryable"))
            response_payload = {
                "ok": False,
                "error": error_message,
                "retryable": retryable,
                "retry_after_seconds": int(resume_result.get("retry_after_seconds") or 0),
                "error_code": str(resume_result.get("error_code") or ""),
            }
            return JsonResponse(response_payload, status=200 if retryable else 400)
        messages.error(request, error_message)
        return redirect(redirect_target)

    if wants_json:
        return JsonResponse(resume_result)

    fetch_meta_for_messages = {
        "rate_limited": bool(resume_result.get("rate_limited")),
        "timed_out": bool(resume_result.get("timed_out")),
        "truncated": bool(resume_result.get("truncated")),
    }
    _add_fetch_meta_messages(
        request,
        fetch_meta_for_messages,
        int(resume_result.get("appended_count") or 0),
        int(resume_result.get("effective_total_tweets") or 0),
        int(resume_result.get("max_total_tweets") or 0),
    )
    appended_count = int(resume_result.get("appended_count") or 0)
    if appended_count > 0:
        messages.success(request, f"Berhasil menambahkan {appended_count} tweet baru ke riwayat.")
    else:
        messages.info(request, "Tidak ada tweet baru yang ditambahkan pada proses lanjutan ini.")

    if bool(resume_result.get("complete")):
        messages.success(request, "Scraping sudah selesai untuk seluruh rentang tanggal.")
    elif history.resume_next_date:
        messages.info(
            request,
            f"Proses belum selesai. Lanjutkan lagi dari {history.resume_next_date.strftime('%d/%m/%Y')}.",
        )

    per_page = _normalize_per_page(request.POST.get("per_page"), DEFAULT_PER_PAGE)
    if next_url:
        return redirect(next_url)
    return redirect(f"{reverse('history_detail', args=[history.id])}?page=1&per_page={per_page}")


@login_required
def prediction_history_detail_view(request: HttpRequest, history_id: int) -> HttpResponse:
    history = get_object_or_404(_prediction_history_queryset_for_detail(request), id=history_id)
    context = _build_prediction_history_view_context(request, history, show_dashboard=False)
    return render(request, "sentiment_app/history_predict_detail.html", context)


@login_required
def prediction_history_dashboard_view(request: HttpRequest, history_id: int) -> HttpResponse:
    history = get_object_or_404(_prediction_history_queryset_for_detail(request), id=history_id)
    context = _build_prediction_history_view_context(request, history, show_dashboard=True)
    return render(request, "sentiment_app/history_predict_detail.html", context)


@login_required
def prediction_history_dashboard_content_view(request: HttpRequest, history_id: int) -> HttpResponse:
    history = get_object_or_404(_prediction_history_queryset_for_detail(request), id=history_id)
    history = _upgrade_prediction_history_scores_if_needed(history)
    rows = history.rows if isinstance(history.rows, list) else []
    source_columns = _normalize_prediction_source_columns(history.columns, rows)

    dashboard = None
    dashboard_error = ""
    if rows:
        try:
            dashboard = _build_prediction_dashboard(history, rows, source_columns)
        except Exception:
            dashboard_error = (
                "Dashboard tidak dapat ditampilkan untuk riwayat prediksi ini. "
                "Silakan periksa data CSV/TXT yang tersimpan."
            )
    else:
        dashboard_error = "Data riwayat CSV/TXT kosong."

    html = render_to_string(
        "sentiment_app/_history_prediction_dashboard_panel.html",
        {
            "dashboard": dashboard,
            "dashboard_error": dashboard_error,
        },
        request=request,
    )
    return JsonResponse({"html": html})


def _build_prediction_history_view_context(
    request: HttpRequest,
    history: PredictionHistory,
    show_dashboard: bool,
) -> dict[str, object]:
    history_table_url = reverse("prediction_history_detail", args=[history.id])
    history_dashboard_url = reverse("prediction_history_dashboard", args=[history.id])
    history_processing = bool(history.is_processing)
    history_error = str(history.error_message or "").strip()
    context: dict[str, object] = {
        "history": history,
        "history_processing": history_processing,
        "history_error": history_error,
        "dashboard_enabled": bool(show_dashboard),
        "show_history_table": not show_dashboard,
        "show_history_dashboard": bool(show_dashboard),
        "history_table_url": history_table_url,
        "history_dashboard_url": history_dashboard_url,
        "history_dashboard_content_url": reverse("prediction_history_dashboard_content", args=[history.id]),
        "history_title": "Dashboard Riwayat CSV/TXT" if show_dashboard else "Detail Riwayat CSV/TXT",
        "dashboard": None,
        "dashboard_error": "",
    }

    if show_dashboard or history_processing or bool(history_error):
        return context

    history = _upgrade_prediction_history_scores_if_needed(history)
    rows = history.rows if isinstance(history.rows, list) else []
    context["history_combined_counts"] = _build_combined_sentiment_counts(rows)

    requested_page = _safe_positive_int(request.GET.get("page"), 1)
    per_page = _normalize_per_page(request.GET.get("per_page"), DEFAULT_PER_PAGE)
    page_rows, total_rows, current_page, total_pages = _paginate_rows(rows, requested_page, per_page)

    source_columns = _normalize_prediction_source_columns(history.columns, rows)
    preview_headers, preview_rows = _build_prediction_history_preview(page_rows, source_columns)
    visible_source_columns = [column for column in source_columns if str(column).strip().lower() != "id"]
    preview_columns = visible_source_columns + PREDICTION_COLUMNS
    preview_source_headers = [PREDICTION_TABLE_HEADERS.get(column, column) for column in preview_columns[: len(visible_source_columns)]]
    probability_column_count = len(PREDICTION_SCORE_COLUMNS)
    preview_probability_headers = [
        PREDICTION_TABLE_HEADERS.get(column, column)
        for column in preview_columns[len(visible_source_columns) : len(visible_source_columns) + probability_column_count]
    ]
    preview_label_headers = [
        PREDICTION_TABLE_HEADERS.get(column, column)
        for column in preview_columns[len(visible_source_columns) + probability_column_count :]
    ]
    preview_text_column = _resolve_prediction_text_column(history.text_column, source_columns)
    preview_text_column_index = preview_columns.index(preview_text_column) if preview_text_column in preview_columns else -1

    page_start = max(1, current_page - 2)
    page_end = min(total_pages, current_page + 2)

    if rows and show_dashboard:
        try:
            context["dashboard"] = _build_prediction_dashboard(history, rows, source_columns)
            context["dashboard_error"] = ""
        except Exception:
            context["dashboard"] = None
            context["dashboard_error"] = (
                "Dashboard tidak dapat ditampilkan untuk riwayat prediksi ini. "
                "Silakan periksa data CSV/TXT yang tersimpan."
            )

    context.update(
        {
            "source_columns": source_columns,
            "batch_preview_text_column_index": preview_text_column_index,
            "batch_preview_source_headers": preview_source_headers,
            "batch_preview_probability_headers": preview_probability_headers,
            "batch_preview_label_headers": preview_label_headers,
            "batch_count": total_rows,
            "batch_preview_headers": preview_headers,
            "batch_preview_rows": preview_rows,
            "current_page": current_page,
            "total_pages": total_pages,
            "page_numbers": range(page_start, page_end + 1),
            "per_page": per_page,
        }
    )

    context["download_url"] = reverse("download_prediction_history_csv", args=[history.id])
    context["download_xlsx_url"] = reverse("download_prediction_history_xlsx", args=[history.id])

    return context


def _build_predict_scraping_context(request: HttpRequest, form: TwitterFetchForm | None = None) -> dict[str, object]:
    dashboard_requested = _is_truthy_flag(request.GET.get("dashboard"))
    dashboard_enabled = bool(dashboard_requested)
    context: dict[str, object] = {
        "scrape_form": form or TwitterFetchForm(),
        "dashboard_enabled": dashboard_enabled,
        "dashboard_query_suffix": "&dashboard=1" if dashboard_enabled else "",
        "dashboard_toggle_url": _build_url_with_dashboard_flag(request),
        "history_mode": False,
    }

    history_id = _safe_positive_int(request.GET.get("history"), 0)
    if history_id:
        history = get_object_or_404(ScrapeHistory, id=history_id, user=request.user)
        dashboard_enabled = bool(dashboard_requested)
        context["dashboard_enabled"] = dashboard_enabled
        context["dashboard_query_suffix"] = "&dashboard=1" if dashboard_enabled else ""
        resume_done_days, resume_total_days, resume_progress_pct = _history_resume_progress(history)
        context["history_mode"] = True
        context["history"] = history
        context["resume_form"] = ResumeScrapeForm()
        context["resume_done_days"] = resume_done_days
        context["resume_total_days"] = resume_total_days
        context["resume_progress_pct"] = resume_progress_pct
        context["resume_next_url"] = request.get_full_path()
        context["auto_resume_default"] = str(request.GET.get("auto", "")).strip() == "1"
        requested_page = _safe_positive_int(request.GET.get("page"), 1)
        per_page = _normalize_per_page(request.GET.get("per_page"), DEFAULT_PER_PAGE)
        if dashboard_enabled:
            rows = _load_scrape_rows(history)
            _apply_scraping_context(
                context,
                rows,
                int(history.tweet_count or 0),
                requested_page,
                per_page,
                history.start_date.isoformat(),
                history.end_date.isoformat(),
                history_id=history.id,
                dashboard_enabled=True,
                query=history.query,
            )
        else:
            page_rows, total_rows, current_page, total_pages = _paginate_scrape_history_rows(
                history,
                requested_page,
                per_page,
            )
            _apply_scraping_page_context(
                context,
                page_rows,
                total_rows,
                current_page,
                total_pages,
                int(history.tweet_count or 0),
                per_page,
                history.start_date.isoformat(),
                history.end_date.isoformat(),
                history_id=history.id,
                dashboard_enabled=False,
            )
        return context

    if request.GET.get("show") != "1":
        request.session.pop(TWITTER_RESULT_SESSION_KEY, None)
        return context

    saved_result = request.session.get(TWITTER_RESULT_SESSION_KEY) or {}
    saved_count = _safe_positive_int(saved_result.get("tweet_count"), 0)
    last_page = _safe_positive_int(saved_result.get("last_page"), 0)
    last_per_page = _normalize_per_page(saved_result.get("last_per_page"), 0)
    saved_history_id = _safe_positive_int(saved_result.get("history_id"), 0)

    if not saved_history_id:
        request.session.pop(TWITTER_RESULT_SESSION_KEY, None)
        return context

    saved_history = ScrapeHistory.objects.filter(id=saved_history_id, user=request.user).only(
        "rows",
        "tweet_count",
        "start_date",
        "end_date",
        "query",
    ).first()
    if saved_history is None:
        request.session.pop(TWITTER_RESULT_SESSION_KEY, None)
        return context

    saved_rows = _load_scrape_rows(saved_history)
    if not saved_rows:
        request.session.pop(TWITTER_RESULT_SESSION_KEY, None)
        return context

    requested_page = _safe_positive_int(request.GET.get("page"), 1)
    per_page = _normalize_per_page(request.GET.get("per_page"), DEFAULT_PER_PAGE)
    if last_page and requested_page == last_page and per_page == last_per_page:
        messages.warning(
            request,
            "Hasil scraping bersifat sementara dan dibersihkan setelah refresh.",
        )
        request.session.pop(TWITTER_RESULT_SESSION_KEY, None)
        return context

    if not _apply_scraping_context(
        context,
        saved_rows,
        saved_count or int(saved_history.tweet_count or 0),
        requested_page,
        per_page,
        saved_history.start_date.isoformat(),
        saved_history.end_date.isoformat(),
        history_id=saved_history.id,
        dashboard_enabled=dashboard_enabled,
        query=saved_history.query,
    ):
        request.session.pop(TWITTER_RESULT_SESSION_KEY, None)
        return context

    saved_result["last_page"] = int(context.get("current_page", 1))
    saved_result["last_per_page"] = per_page
    request.session[TWITTER_RESULT_SESSION_KEY] = saved_result
    request.session.modified = True
    return context


@login_required
def predict_view(request: HttpRequest) -> HttpResponse:
    submitted_mode = ""
    if request.method == "POST":
        submitted_mode = request.POST.get("input_mode", "").strip().lower()
        if submitted_mode == "scraping":
            return twitter_fetch_view(request)

    form = PredictForm(request.POST or None, request.FILES or None)
    scrape_form = TwitterFetchForm()
    requested_tab = str(request.GET.get("tab") or "").strip().lower()
    active_tab = "scraping"
    if requested_tab in {"file", "scraping"}:
        active_tab = requested_tab
    if request.GET.get("show") == "1" or request.GET.get("history"):
        active_tab = "scraping"
    if submitted_mode == "file":
        active_tab = submitted_mode

    selected_model_version = ""
    if form.is_bound:
        selected_model_version = str(form.data.get("model_version") or "").strip()
    if not selected_model_version:
        model_version_choices = list(form.fields["model_version"].choices)
        if model_version_choices:
            selected_model_version = str(model_version_choices[0][0] or "").strip()

    context: dict[str, object] = {
        "form": form,
        "scrape_form": scrape_form,
        "active_tab": active_tab,
        "selected_model_version": selected_model_version,
    }
    context.update(_build_predict_scraping_context(request, form=scrape_form))

    if request.method == "POST" and active_tab != "scraping" and form.is_valid():
        upload_file = form.cleaned_data.get("upload_file")
        text_column = form.cleaned_data.get("text_column")
        model_version = resolve_model_version_name(form.cleaned_data.get("model_version"))

        try:
            if upload_file:
                upload_file.seek(0)
                uploaded_content = upload_file.read()
                saved_history = PredictionHistory.objects.create(
                    user=request.user,
                    source_name=getattr(upload_file, "name", "") or "",
                    model_version=model_version,
                    text_column=(text_column or "").strip(),
                    sample_count=0,
                    columns=[],
                    rows=[],
                    is_processing=True,
                    error_message="",
                )
                _launch_background_history_job(
                    f"prediction-history-{saved_history.id}",
                    _process_prediction_history_file_job,
                    saved_history.id,
                    str(getattr(upload_file, "name", "") or ""),
                    bytes(uploaded_content),
                    str(getattr(upload_file, "content_type", "") or ""),
                    str(text_column or "").strip(),
                    model_version,
                )
                return redirect("prediction_history_detail", history_id=saved_history.id)
        except (ModelServiceError, FileValidationError) as exc:
            messages.error(request, str(exc))
        except Exception as exc:
            messages.error(request, f"Terjadi kesalahan tak terduga saat menjalankan prediksi: {exc}")

    return render(request, "sentiment_app/predict.html", context)


@login_required
def beranda_view(request: HttpRequest) -> HttpResponse:
    scrape_histories = ScrapeHistory.objects.filter(user=request.user)
    prediction_histories = _user_prediction_dataset_queryset(request.user)
    total_scraping_results = (
        scrape_histories.aggregate(total=models.Sum("tweet_count")).get("total") or 0
    )
    total_prediction_results = (
        prediction_histories.aggregate(total=models.Sum("sample_count")).get("total") or 0
    )
    context = {
        "total_scraping_count": scrape_histories.count(),
        "total_scraping_results": total_scraping_results,
        "total_prediction_count": prediction_histories.count(),
        "total_prediction_results": total_prediction_results,
    }
    return render(request, "sentiment_app/beranda.html", context)


@login_required
def twitter_fetch_view(request: HttpRequest) -> HttpResponse:
    predict_scraping_base = f"{reverse('predict')}?tab=scraping"
    if request.method == "POST":
        form = TwitterFetchForm(request.POST)
        if not form.is_valid():
            for message in form.non_field_errors():
                messages.error(request, str(message))
            for field_name, field_errors in form.errors.items():
                if field_name == "__all__":
                    continue
                label = form.fields.get(field_name).label if field_name in form.fields else field_name
                for field_error in field_errors:
                    messages.error(request, f"{label}: {field_error}")
            return redirect(predict_scraping_base)

        api_key = (form.cleaned_data.get("api_key") or "").strip()
        model_version = resolve_model_version_name(form.cleaned_data.get("model_version"))
        query = form.cleaned_data.get("query")
        language = form.cleaned_data.get("language")
        start_date = form.cleaned_data.get("start_date")
        end_date = form.cleaned_data.get("end_date")
        runtime_config = _build_scrape_runtime_config(start_date, end_date)

        if not api_key:
            messages.error(request, "API key wajib diisi.")
            return redirect(predict_scraping_base)

        try:
            history_item = ScrapeHistory.objects.create(
                user=request.user,
                query=str(query or ""),
                model_version=model_version,
                language=str(language or ""),
                start_date=start_date,
                end_date=end_date,
                tweet_count=0,
                rows=[],
                is_complete=False,
                is_processing=False,
                resume_next_date=start_date,
                stop_reason="processing",
                error_message="",
            )
            return redirect(f"{reverse('history_detail', args=[history_item.id])}?auto=1")
        except (TwitterAPIError, ModelServiceError, FileValidationError) as exc:
            messages.error(request, str(exc))
            return redirect(predict_scraping_base)
        except Exception as exc:
            messages.error(request, f"Terjadi kesalahan tak terduga saat mengambil/mengklasifikasikan tweet: {exc}")
            return redirect(predict_scraping_base)

    return redirect(predict_scraping_base)


@login_required
def download_prediction_history_csv_view(request: HttpRequest, history_id: int) -> HttpResponse:
    history = get_object_or_404(_prediction_history_queryset_for_detail(request), id=history_id)
    history = _upgrade_prediction_history_scores_if_needed(history)

    rows = [row for row in (history.rows if isinstance(history.rows, list) else []) if isinstance(row, dict)]
    if not rows:
        raise Http404("Data prediksi tidak ditemukan.")

    source_columns = _normalize_prediction_source_columns(history.columns, rows)
    export_columns = source_columns + PREDICTION_COLUMNS
    export_headers = [PREDICTION_HEADERS.get(column, column) for column in export_columns]
    filename = _safe_download_filename(history.source_name or f"prediction_history_{history.id}", "prediction_history")
    return _csv_download_response(filename, export_columns, export_headers, rows)


@login_required
def download_prediction_history_xlsx_view(request: HttpRequest, history_id: int) -> HttpResponse:
    history = get_object_or_404(_prediction_history_queryset_for_detail(request), id=history_id)
    history = _upgrade_prediction_history_scores_if_needed(history)

    rows = [row for row in (history.rows if isinstance(history.rows, list) else []) if isinstance(row, dict)]
    if not rows:
        raise Http404("Data prediksi tidak ditemukan.")

    source_columns = _normalize_prediction_source_columns(history.columns, rows)
    export_columns = source_columns + PREDICTION_COLUMNS
    export_headers = [PREDICTION_HEADERS.get(column, column) for column in export_columns]
    filename = _safe_download_filename(
        history.source_name or f"prediction_history_{history.id}",
        "prediction_history",
        extension=".xlsx",
    )
    return _xlsx_download_response(filename, export_columns, export_headers, rows)


@login_required
def download_scrape_history_csv_view(request: HttpRequest, history_id: int) -> HttpResponse:
    history = get_object_or_404(_scrape_history_queryset_for_detail(request), id=history_id)
    history = _upgrade_scrape_history_scores_if_needed(history)

    rows = _load_scrape_rows(history)
    if not rows:
        raise Http404("Data scraping tidak ditemukan.")

    export_headers = [SCRAPE_HEADERS.get(column, column) for column in SCRAPE_EXPORT_COLUMNS]
    filename = _safe_download_filename(f"scrape_history_{history.id}", "scrape_history")
    return _csv_download_response(filename, SCRAPE_EXPORT_COLUMNS, export_headers, rows)


@login_required
def download_scrape_history_xlsx_view(request: HttpRequest, history_id: int) -> HttpResponse:
    history = get_object_or_404(_scrape_history_queryset_for_detail(request), id=history_id)
    history = _upgrade_scrape_history_scores_if_needed(history)

    rows = _load_scrape_rows(history)
    if not rows:
        raise Http404("Data scraping tidak ditemukan.")

    export_headers = [SCRAPE_HEADERS.get(column, column) for column in SCRAPE_EXPORT_COLUMNS]
    filename = _safe_download_filename(
        f"scrape_history_{history.id}",
        "scrape_history",
        extension=".xlsx",
    )
    return _xlsx_download_response(filename, SCRAPE_EXPORT_COLUMNS, export_headers, rows)
