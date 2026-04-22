from __future__ import annotations

import base64
import io
import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from django.conf import settings
from django.contrib.auth import login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator
from django.db import models
from django.http import FileResponse, Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from .forms import LoginForm, PredictForm, ResumeScrapeForm, TwitterFetchForm
from .models import PredictionHistory, ScrapeHistory, ScrapeTempChunk
from .services.file_service import (
    FileValidationError,
    generate_classification_csv,
    parse_uploaded_file,
)
from .services.model_service import ModelServiceError, predict_batch, predict_batch_in_chunks, predict_single
from .services.preprocess import preprocess_text
from .services.twitter_client import TwitterAPIError, TwitterRateLimitError, TwitterTimeoutError, fetch_tweets

try:
    from wordcloud import STOPWORDS as WORDCLOUD_BASE_STOPWORDS
    from wordcloud import WordCloud
except Exception:
    WORDCLOUD_BASE_STOPWORDS = set()
    WordCloud = None

SAFE_OUTPUT_RE = re.compile(r"^[A-Za-z0-9_-]+\.csv$")
DEFAULT_PER_PAGE = 10
MAX_PER_PAGE = 200
HISTORY_PER_PAGE = 10
TWITTER_RESULT_SESSION_KEY = "twitter_last_result"
PREDICTION_COLUMNS = ["knn_label", "knn_score", "svm_label", "svm_score"]
PREDICTION_HEADERS = {
    "knn_label": "KNN",
    "knn_score": "Skor KNN (0-1)",
    "svm_label": "SVM",
    "svm_score": "Skor SVM (-1 s/d 1)",
}
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
WORDCLOUD_STOPWORDS = {
    "dan",
    "atau",
    "yang",
    "untuk",
    "dengan",
    "pada",
    "dari",
    "ke",
    "di",
    "ini",
    "itu",
    "kamu",
    "kami",
    "kita",
    "mereka",
    "saya",
    "aku",
    "nya",
    "aja",
    "juga",
    "udah",
    "sudah",
    "belum",
    "nih",
    "ya",
    "yah",
    "deh",
    "dong",
    "lagi",
    "jadi",
    "karena",
    "agar",
    "buat",
    "lebih",
    "dalam",
    "about",
    "after",
    "again",
    "all",
    "also",
    "and",
    "are",
    "but",
    "for",
    "from",
    "has",
    "have",
    "his",
    "her",
    "him",
    "how",
    "its",
    "just",
    "like",
    "not",
    "our",
    "out",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "too",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "you",
    "your",
}


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

    page_rows: list[dict[str, object]] = []
    for idx, row in enumerate(page_slice, start=1):
        normalized_row = dict(row)
        normalized_row["row_number"] = start_idx + idx
        page_rows.append(normalized_row)

    return page_rows, total_rows, current_page, total_pages


def _build_batch_preview(
    source_rows: list[dict[str, str]],
    source_columns: list[str],
    predictions: list[dict[str, object]],
) -> tuple[list[str], list[dict[str, object]]]:
    columns = list(source_columns) + PREDICTION_COLUMNS
    headers = [PREDICTION_HEADERS.get(column, column) for column in columns]

    preview_rows: list[dict[str, object]] = []
    for index, (source_row, prediction) in enumerate(zip(source_rows, predictions), start=1):
        merged = {
            **source_row,
            "knn_label": prediction.get("knn_label", ""),
            "knn_score": prediction.get("knn_score"),
            "svm_label": prediction.get("svm_label", ""),
            "svm_score": prediction.get("svm_score"),
        }

        cells = []
        for column in columns:
            cell_type = "text"
            if column in ("knn_label", "svm_label"):
                cell_type = "label"
            elif column in ("knn_score", "svm_score"):
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
                "knn_score": prediction.get("knn_score"),
                "svm_label": prediction.get("svm_label", ""),
                "svm_score": prediction.get("svm_score"),
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
        if key not in PREDICTION_COLUMNS and key != "row_number":
            derived_columns.append(str(key))
    return derived_columns


def _build_prediction_history_preview(
    rows: list[dict[str, object]],
    source_columns: list[str],
) -> tuple[list[str], list[dict[str, object]]]:
    columns = list(source_columns) + PREDICTION_COLUMNS
    headers = [PREDICTION_HEADERS.get(column, column) for column in columns]
    preview_rows: list[dict[str, object]] = []

    for row in rows:
        index = int(row.get("row_number", len(preview_rows) + 1))
        cells: list[dict[str, object]] = []
        for column in columns:
            cell_type = "text"
            if column in ("knn_label", "svm_label"):
                cell_type = "label"
            elif column in ("knn_score", "svm_score"):
                cell_type = "score"
            cells.append(
                {
                    "type": cell_type,
                    "value": row.get(column, ""),
                }
            )
        preview_rows.append({"index": index, "cells": cells})

    return headers, preview_rows


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


def _clean_text_for_wordcloud(text: str) -> str:
    # Align with model preprocessing so WordCloud reflects the same cleaned language space.
    cleaned = preprocess_text(str(text or ""), apply_stemming=False)
    cleaned = re.sub(r"[^0-9a-zA-Z_\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _build_wordcloud_image(texts: list[str], colormap: str) -> str | None:
    if WordCloud is None:
        return None

    bigram_counter: Counter[str] = Counter()
    for raw_text in texts:
        cleaned_text = _clean_text_for_wordcloud(str(raw_text or ""))
        if not cleaned_text:
            continue
        tokens = [token for token in cleaned_text.split() if token]
        if len(tokens) < 2:
            continue
        for idx in range(len(tokens) - 1):
            bigram_counter[f"{tokens[idx]} {tokens[idx + 1]}"] += 1

    if not bigram_counter:
        return None

    stopwords = set(WORDCLOUD_BASE_STOPWORDS) | WORDCLOUD_STOPWORDS
    cloud = WordCloud(
        width=1400,
        height=800,
        background_color="white",
        stopwords=stopwords,
        collocations=False,
        max_words=140,
        margin=24,
        min_font_size=10,
        max_font_size=96,
        relative_scaling=0.35,
        prefer_horizontal=1.0,
        colormap=colormap,
    ).generate_from_frequencies(bigram_counter)

    buffer = io.BytesIO()
    cloud.to_image().save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _choose_granularity(start_date: date, end_date: date) -> tuple[str, str]:
    total_days = max(1, (end_date - start_date).days)
    if total_days <= 14:
        return "day", "Harian"
    if total_days < 60:
        return "week", "Mingguan"
    if total_days <= 61:
        return "week", "Mingguan"
    return "month", "Bulanan"


def _build_scraping_dashboard(
    rows: list[dict[str, object]],
    start_date: date | None = None,
    end_date: date | None = None,
    trend_subject: str = "Tweet",
) -> dict[str, object]:
    max_texts_per_label = _setting_positive_int("SENTIMENT_WORDCLOUD_MAX_TEXTS_PER_LABEL", 1200)
    max_chars_per_label = _setting_positive_int("SENTIMENT_WORDCLOUD_MAX_CHARS_PER_LABEL", 160000)

    knn_positive_texts: list[str] = []
    knn_negative_texts: list[str] = []
    svm_positive_texts: list[str] = []
    svm_negative_texts: list[str] = []
    knn_positive_chars = 0
    knn_negative_chars = 0
    svm_positive_chars = 0
    svm_negative_chars = 0
    knn_counts: Counter[str] = Counter()
    svm_counts: Counter[str] = Counter()
    row_dates: list[date] = []

    for row in rows:
        text = str(row.get("text", "") or "")

        knn_label = _normalize_sentiment_label(row.get("knn_label"))
        svm_label = _normalize_sentiment_label(row.get("svm_label"))
        knn_counts[knn_label] += 1
        svm_counts[svm_label] += 1

        if text:
            if knn_label == "Positive":
                if len(knn_positive_texts) < max_texts_per_label and knn_positive_chars < max_chars_per_label:
                    clipped_text = text[: max(1, max_chars_per_label - knn_positive_chars)]
                    knn_positive_chars += len(clipped_text)
                    knn_positive_texts.append(clipped_text)
            elif knn_label == "Negative":
                if len(knn_negative_texts) < max_texts_per_label and knn_negative_chars < max_chars_per_label:
                    clipped_text = text[: max(1, max_chars_per_label - knn_negative_chars)]
                    knn_negative_chars += len(clipped_text)
                    knn_negative_texts.append(clipped_text)

            if svm_label == "Positive":
                if len(svm_positive_texts) < max_texts_per_label and svm_positive_chars < max_chars_per_label:
                    clipped_text = text[: max(1, max_chars_per_label - svm_positive_chars)]
                    svm_positive_chars += len(clipped_text)
                    svm_positive_texts.append(clipped_text)
            elif svm_label == "Negative":
                if len(svm_negative_texts) < max_texts_per_label and svm_negative_chars < max_chars_per_label:
                    clipped_text = text[: max(1, max_chars_per_label - svm_negative_chars)]
                    svm_negative_chars += len(clipped_text)
                    svm_negative_texts.append(clipped_text)

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
        "knn_positive_image": None,
        "knn_negative_image": None,
        "svm_positive_image": None,
        "svm_negative_image": None,
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
            wordclouds["knn_positive_image"] = _build_wordcloud_image(knn_positive_texts, colormap="Greens")
            wordclouds["knn_negative_image"] = _build_wordcloud_image(knn_negative_texts, colormap="Reds")
            wordclouds["svm_positive_image"] = _build_wordcloud_image(svm_positive_texts, colormap="Greens")
            wordclouds["svm_negative_image"] = _build_wordcloud_image(svm_negative_texts, colormap="Reds")
        except Exception as exc:
            wordcloud_error = f"Gagal membuat WordCloud: {exc}"

    return {
        "period_label": f"{start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}",
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
        "wordcloud_available": WordCloud is not None and not wordcloud_error,
        "wordcloud_error": wordcloud_error,
        "wordclouds": wordclouds,
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
        if key in PREDICTION_COLUMNS or key == "row_number":
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
        if key in PREDICTION_COLUMNS or key == "row_number":
            continue
        if not any(hint in normalized_key for hint in PREDICTION_DATE_COLUMN_HINTS):
            continue
        if _parse_created_at_date(value) is not None:
            return value

    return ""


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
    dashboard_enabled: bool = False,
) -> bool:
    page_rows, total_rows, current_page, total_pages = _paginate_rows(rows, requested_page, per_page)
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
    if not dashboard_enabled:
        context["dashboard"] = None
        context["dashboard_error"] = ""
        return True

    try:
        context["dashboard"] = _build_scraping_dashboard(
            rows,
            _safe_parse_iso_date(start_date_value),
            _safe_parse_iso_date(end_date_value),
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
                "knn_score": prediction.get("knn_score"),
                "svm_label": prediction.get("svm_label", ""),
                "svm_score": prediction.get("svm_score"),
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
    window_days: int,
) -> None:
    next_start_date = _next_fetch_start_date(fetch_meta, fallback_start_date)
    timed_out = bool(fetch_meta.get("timed_out"))
    rate_limited = bool(fetch_meta.get("rate_limited"))
    truncated = bool(fetch_meta.get("truncated"))
    is_finished = next_start_date > end_date and not (timed_out or rate_limited or truncated)

    history.window_days = max(1, int(window_days))
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
    if selected_days <= 7:
        window_days = 1
    elif selected_days <= 31:
        window_days = 2
    elif selected_days <= 90:
        window_days = 3
    else:
        window_days = 4

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

    runtime_config = _build_scrape_runtime_config(resume_start_date, history.end_date)
    effective_total_tweets = int(runtime_config["effective_total_tweets"])
    effective_tweets_per_window = int(runtime_config["effective_tweets_per_window"])
    max_total_tweets = int(runtime_config["max_total_tweets"])
    max_runtime_seconds = int(runtime_config["max_runtime_seconds"])
    predict_chunk_size = int(runtime_config["predict_chunk_size"])
    default_window_days = int(runtime_config["window_days"])
    window_days = int(history.window_days or default_window_days)
    window_days = max(1, window_days)

    seen_keys = _build_existing_scrape_keys(history)
    chunk_index = _next_chunk_index(history)
    appended_count = 0

    def _handle_window(window_tweets: list[dict[str, object]]) -> None:
        nonlocal chunk_index, appended_count
        if not window_tweets:
            return

        predictions = predict_batch_in_chunks(
            [str(tweet.get("text", "")) for tweet in window_tweets],
            chunk_size=predict_chunk_size,
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

    try:
        fetch_result = fetch_tweets(
            api_key=api_key,
            query=history.query,
            language=history.language,
            start_date=resume_start_date.isoformat(),
            end_date=history.end_date.isoformat(),
            max_tweets_per_window=effective_tweets_per_window,
            max_total_tweets=effective_total_tweets,
            window_days=window_days,
            max_runtime_seconds=max_runtime_seconds,
            on_window=_handle_window,
            return_meta=True,
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
        "next_start_date": resume_start_date.isoformat(),
    }
    if isinstance(fetch_result, tuple) and len(fetch_result) == 2 and isinstance(fetch_result[1], dict):
        fetch_meta = fetch_result[1]

    total_rows = _load_scrape_rows(history)
    history.tweet_count = len(total_rows)
    _apply_history_progress_meta(history, fetch_meta, history.end_date, resume_start_date, window_days)
    history.save(update_fields=["tweet_count", "is_complete", "resume_next_date", "stop_reason", "window_days"])
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
    prediction_histories_qs = PredictionHistory.objects.filter(user=request.user).only(
        "id",
        "input_type",
        "source_name",
        "text_column",
        "sample_count",
        "created_at",
    )

    scrape_page_number = _safe_positive_int(request.GET.get("scrape_page"), 1)
    prediction_page_number = _safe_positive_int(request.GET.get("pred_page"), 1)

    scrape_page_obj = Paginator(scrape_histories_qs, HISTORY_PER_PAGE).get_page(scrape_page_number)
    prediction_page_obj = Paginator(prediction_histories_qs, HISTORY_PER_PAGE).get_page(prediction_page_number)

    context = {
        "scrape_histories": scrape_page_obj.object_list,
        "prediction_histories": prediction_page_obj.object_list,
        "scrape_page_obj": scrape_page_obj,
        "prediction_page_obj": prediction_page_obj,
    }
    return render(request, "sentiment_app/history.html", context)


def _history_list_redirect(request: HttpRequest) -> HttpResponse:
    scrape_page = _safe_positive_int(request.POST.get("scrape_page"), 1)
    pred_page = _safe_positive_int(request.POST.get("pred_page"), 1)
    return redirect(f"{reverse('history_list')}?scrape_page={scrape_page}&pred_page={pred_page}")


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
        queryset = PredictionHistory.objects.filter(user=request.user)
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
    history = get_object_or_404(ScrapeHistory, id=history_id, user=request.user)
    form = TwitterFetchForm()
    resume_form = ResumeScrapeForm()
    resume_done_days, resume_total_days, resume_progress_pct = _history_resume_progress(history)
    requested_page = _safe_positive_int(request.GET.get("page"), 1)
    per_page = _normalize_per_page(request.GET.get("per_page"), DEFAULT_PER_PAGE)
    dashboard_requested = _is_truthy_flag(request.GET.get("dashboard"))
    dashboard_enabled = bool(dashboard_requested or history.is_complete)
    rows = _load_scrape_rows(history)

    context: dict[str, object] = {
        "form": form,
        "resume_form": resume_form,
        "history_mode": True,
        "history": history,
        "resume_done_days": resume_done_days,
        "resume_total_days": resume_total_days,
        "resume_progress_pct": resume_progress_pct,
        "resume_next_url": request.get_full_path(),
        "auto_resume_default": str(request.GET.get("auto", "")).strip() == "1",
        "dashboard_enabled": dashboard_enabled,
        "dashboard_query_suffix": "&dashboard=1" if dashboard_enabled else "",
        "dashboard_toggle_url": _build_url_with_dashboard_flag(request),
    }
    if not _apply_scraping_context(
        context,
        rows,
        int(history.tweet_count or 0),
        requested_page,
        per_page,
        history.start_date.isoformat(),
        history.end_date.isoformat(),
        dashboard_enabled=dashboard_enabled,
    ):
        messages.warning(request, "Data riwayat scraping kosong.")

    return render(request, "sentiment_app/twitter.html", context)


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
    history = get_object_or_404(PredictionHistory, id=history_id, user=request.user)
    context: dict[str, object] = {
        "history": history,
    }

    rows = history.rows if isinstance(history.rows, list) else []

    if history.input_type == PredictionHistory.InputType.SINGLE:
        single_row: dict[str, object] = {}
        if rows and isinstance(rows[0], dict):
            single_row = rows[0]
        context["single_result"] = {
            "text": single_row.get("text", history.text_input),
            "knn_label": single_row.get("knn_label", ""),
            "knn_score": single_row.get("knn_score"),
            "svm_label": single_row.get("svm_label", ""),
            "svm_score": single_row.get("svm_score"),
        }
        return render(request, "sentiment_app/history_predict_detail.html", context)

    requested_page = _safe_positive_int(request.GET.get("page"), 1)
    per_page = _normalize_per_page(request.GET.get("per_page"), DEFAULT_PER_PAGE)
    page_rows, total_rows, current_page, total_pages = _paginate_rows(rows, requested_page, per_page)

    source_columns = _normalize_prediction_source_columns(history.columns, rows)
    preview_headers, preview_rows = _build_prediction_history_preview(page_rows, source_columns)

    page_start = max(1, current_page - 2)
    page_end = min(total_pages, current_page + 2)

    if rows:
        context["dashboard_enabled"] = True
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
            "batch_count": total_rows,
            "batch_preview_headers": preview_headers,
            "batch_preview_rows": preview_rows,
            "current_page": current_page,
            "total_pages": total_pages,
            "page_numbers": range(page_start, page_end + 1),
            "per_page": per_page,
        }
    )

    if history.output_filename:
        context["download_url"] = reverse("download_output", args=[history.output_filename])

    return render(request, "sentiment_app/history_predict_detail.html", context)


@login_required
def predict_view(request: HttpRequest) -> HttpResponse:
    form = PredictForm(request.POST or None, request.FILES or None)
    active_tab = "single"
    if request.method == "POST":
        submitted_mode = request.POST.get("input_mode", "").strip().lower()
        if submitted_mode in {"single", "file"}:
            active_tab = submitted_mode

    context: dict[str, object] = {
        "form": form,
        "active_tab": active_tab,
    }

    if request.method == "POST" and form.is_valid():
        text_input = form.cleaned_data.get("text_input")
        upload_file = form.cleaned_data.get("upload_file")
        text_column = form.cleaned_data.get("text_column")

        try:
            if text_input:
                single_result = predict_single(text_input)
                context["single_result"] = single_result
                context["active_tab"] = "single"
                PredictionHistory.objects.create(
                    user=request.user,
                    input_type=PredictionHistory.InputType.SINGLE,
                    text_input=text_input,
                    sample_count=1,
                    rows=_serialize_history_rows(
                        [
                            {
                                "text": single_result.get("text", text_input),
                                "knn_label": single_result.get("knn_label", ""),
                                "knn_score": single_result.get("knn_score"),
                                "svm_label": single_result.get("svm_label", ""),
                                "svm_score": single_result.get("svm_score"),
                            }
                        ]
                    ),
                )
            elif upload_file:
                texts, detected_column, source_rows, source_columns = parse_uploaded_file(upload_file, text_column)
                predictions = predict_batch(texts)
                output_filename = generate_classification_csv(predictions, prefix="uploaded")
                preview_headers, preview_rows = _build_batch_preview(source_rows, source_columns, predictions)
                merged_rows = _merge_batch_rows_for_history(source_rows, predictions)
                context["batch_count"] = len(predictions)
                context["detected_column"] = detected_column
                context["batch_preview_headers"] = preview_headers
                context["batch_preview_rows"] = preview_rows[:20]
                context["output_filename"] = output_filename
                context["download_url"] = reverse("download_output", args=[output_filename])
                context["active_tab"] = "file"
                PredictionHistory.objects.create(
                    user=request.user,
                    input_type=PredictionHistory.InputType.FILE,
                    source_name=getattr(upload_file, "name", "") or "",
                    text_column=(detected_column or text_column or "").strip(),
                    sample_count=len(merged_rows),
                    columns=[str(column) for column in source_columns],
                    rows=_serialize_history_rows(merged_rows),
                    output_filename=output_filename,
                )
        except (ModelServiceError, FileValidationError) as exc:
            messages.error(request, str(exc))
        except Exception as exc:
            messages.error(request, f"Terjadi kesalahan tak terduga saat menjalankan prediksi: {exc}")

    return render(request, "sentiment_app/predict.html", context)


@login_required
def beranda_view(request: HttpRequest) -> HttpResponse:
    return render(request, "sentiment_app/beranda.html")


@login_required
def twitter_fetch_view(request: HttpRequest) -> HttpResponse:
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
            return redirect("twitter_fetch")

        api_key = (form.cleaned_data.get("api_key") or "").strip()
        query = form.cleaned_data.get("query")
        language = form.cleaned_data.get("language")
        per_page = _normalize_per_page(
            request.GET.get("per_page"),
            DEFAULT_PER_PAGE,
        )
        start_date = form.cleaned_data.get("start_date")
        end_date = form.cleaned_data.get("end_date")
        runtime_config = _build_scrape_runtime_config(start_date, end_date)
        use_temp_db_mode = bool(runtime_config["use_temp_db_mode"])
        window_days = int(runtime_config["window_days"])
        effective_total_tweets = int(runtime_config["effective_total_tweets"])
        effective_tweets_per_window = int(runtime_config["effective_tweets_per_window"])
        max_total_tweets = int(runtime_config["max_total_tweets"])
        max_runtime_seconds = int(runtime_config["max_runtime_seconds"])
        predict_chunk_size = int(runtime_config["predict_chunk_size"])

        if not api_key:
            messages.error(request, "API key wajib diisi.")
            return redirect("twitter_fetch")

        try:
            if use_temp_db_mode:
                history_item = ScrapeHistory.objects.create(
                    user=request.user,
                    query=str(query or ""),
                    language=str(language or ""),
                    start_date=start_date,
                    end_date=end_date,
                    tweet_count=0,
                    rows=[],
                    is_complete=False,
                    resume_next_date=start_date,
                    stop_reason="partial",
                    window_days=window_days,
                )
                temp_rows_count = 0
                temp_chunk_index = _next_chunk_index(history_item)
                seen_keys: set[str] = set()

                def _handle_window(window_tweets: list[dict[str, object]]) -> None:
                    nonlocal temp_rows_count, temp_chunk_index
                    if not window_tweets:
                        return

                    predictions = predict_batch_in_chunks(
                        [str(tweet.get("text", "")) for tweet in window_tweets],
                        chunk_size=predict_chunk_size,
                    )
                    classified_rows = _combine_tweet_predictions(window_tweets, predictions)
                    filtered_rows = _filter_rows_by_date_range(classified_rows, start_date, end_date)
                    if not filtered_rows:
                        return

                    appended_count, next_chunk_index = _append_rows_to_history_chunks(
                        history_item,
                        filtered_rows,
                        seen_keys,
                        temp_chunk_index,
                    )
                    temp_chunk_index = next_chunk_index
                    temp_rows_count += appended_count

                fetch_result = fetch_tweets(
                    api_key=api_key,
                    query=query,
                    language=language,
                    start_date=start_date.isoformat() if start_date else None,
                    end_date=end_date.isoformat() if end_date else None,
                    max_tweets_per_window=effective_tweets_per_window,
                    max_total_tweets=effective_total_tweets,
                    window_days=window_days,
                    max_runtime_seconds=max_runtime_seconds,
                    on_window=_handle_window,
                    return_meta=True,
                )
                fetch_meta: dict[str, object] = {
                    "rate_limited": False,
                    "truncated": False,
                    "timed_out": False,
                    "next_start_date": start_date.isoformat(),
                }
                if isinstance(fetch_result, tuple) and len(fetch_result) == 2 and isinstance(fetch_result[1], dict):
                    fetch_meta = fetch_result[1]

                if temp_rows_count <= 0:
                    history_item.delete()
                    messages.warning(
                        request,
                        "Tidak ada tweet dalam rentang tanggal yang dipilih setelah validasi tanggal final.",
                    )
                    request.session.pop(TWITTER_RESULT_SESSION_KEY, None)
                    return redirect("twitter_fetch")

                history_item.tweet_count = temp_rows_count
                _apply_history_progress_meta(history_item, fetch_meta, end_date, start_date, window_days)
                history_item.save(
                    update_fields=["tweet_count", "is_complete", "resume_next_date", "stop_reason", "window_days"]
                )

                _add_fetch_meta_messages(
                    request,
                    fetch_meta,
                    temp_rows_count,
                    effective_total_tweets,
                    max_total_tweets,
                )
                messages.info(
                    request,
                    "Rentang scraping lebih dari 3 bulan diproses menggunakan penyimpanan sementara di database.",
                )

                request.session[TWITTER_RESULT_SESSION_KEY] = {
                    "history_id": history_item.id,
                    "tweet_count": temp_rows_count,
                    "last_page": 0,
                    "last_per_page": 0,
                }
                request.session.modified = True
                query_url = reverse("twitter_fetch")
                auto_flag = "1" if not history_item.is_complete else "0"
                return redirect(
                    f"{query_url}?show=1&history={history_item.id}&page=1&per_page={per_page}&auto={auto_flag}"
                )

            fetch_result = fetch_tweets(
                api_key=api_key,
                query=query,
                language=language,
                start_date=start_date.isoformat() if start_date else None,
                end_date=end_date.isoformat() if end_date else None,
                max_tweets_per_window=effective_tweets_per_window,
                max_total_tweets=effective_total_tweets,
                window_days=window_days,
                max_runtime_seconds=max_runtime_seconds,
                return_meta=True,
            )
            fetch_meta: dict[str, object] = {
                "rate_limited": False,
                "truncated": False,
                "timed_out": False,
                "next_start_date": start_date.isoformat(),
            }
            if isinstance(fetch_result, tuple) and len(fetch_result) == 2 and isinstance(fetch_result[1], dict):
                tweets = fetch_result[0]
                fetch_meta = fetch_result[1]
            else:
                tweets = fetch_result  # type: ignore[assignment]
            if not tweets:
                messages.warning(request, "Tidak ada tweet yang ditemukan untuk permintaan ini.")
                request.session.pop(TWITTER_RESULT_SESSION_KEY, None)
                return redirect("twitter_fetch")

            predictions = predict_batch_in_chunks(
                [str(tweet.get("text", "")) for tweet in tweets],
                chunk_size=predict_chunk_size,
            )
            classified_rows = _combine_tweet_predictions(tweets, predictions)
            filtered_rows = _filter_rows_by_date_range(classified_rows, start_date, end_date)

            if not filtered_rows:
                messages.warning(
                    request,
                    "Tidak ada tweet dalam rentang tanggal yang dipilih setelah validasi tanggal final.",
                )
                request.session.pop(TWITTER_RESULT_SESSION_KEY, None)
                return redirect("twitter_fetch")

            history_rows = _serialize_history_rows(filtered_rows)
            history_item = ScrapeHistory.objects.create(
                user=request.user,
                query=str(query or ""),
                language=str(language or ""),
                start_date=start_date,
                end_date=end_date,
                tweet_count=len(history_rows),
                rows=history_rows,
                is_complete=True,
                resume_next_date=None,
                stop_reason="",
                window_days=window_days,
            )
            _apply_history_progress_meta(history_item, fetch_meta, end_date, start_date, window_days)
            history_item.tweet_count = len(history_rows)
            history_item.save(update_fields=["tweet_count", "is_complete", "resume_next_date", "stop_reason", "window_days"])

            _add_fetch_meta_messages(
                request,
                fetch_meta,
                len(history_rows),
                effective_total_tweets,
                max_total_tweets,
            )

            request.session[TWITTER_RESULT_SESSION_KEY] = {
                "history_id": history_item.id,
                "tweet_count": len(history_rows),
                "last_page": 0,
                "last_per_page": 0,
            }
            request.session.modified = True

            query_url = reverse("twitter_fetch")
            auto_flag = "1" if not history_item.is_complete else "0"
            return redirect(f"{query_url}?show=1&history={history_item.id}&page=1&per_page={per_page}&auto={auto_flag}")
        except (TwitterAPIError, ModelServiceError, FileValidationError) as exc:
            messages.error(request, str(exc))
            return redirect("twitter_fetch")
        except Exception as exc:
            messages.error(request, f"Terjadi kesalahan tak terduga saat mengambil/mengklasifikasikan tweet: {exc}")
            return redirect("twitter_fetch")

    form = TwitterFetchForm()
    dashboard_requested = _is_truthy_flag(request.GET.get("dashboard"))
    dashboard_enabled = bool(dashboard_requested)
    context: dict[str, object] = {
        "form": form,
        "dashboard_enabled": dashboard_enabled,
        "dashboard_query_suffix": "&dashboard=1" if dashboard_enabled else "",
        "dashboard_toggle_url": _build_url_with_dashboard_flag(request),
    }

    history_id = _safe_positive_int(request.GET.get("history"), 0)
    if history_id:
        history = get_object_or_404(ScrapeHistory, id=history_id, user=request.user)
        dashboard_enabled = bool(dashboard_requested or history.is_complete)
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
        rows = _load_scrape_rows(history)
        _apply_scraping_context(
            context,
            rows,
            int(history.tweet_count or 0),
            requested_page,
            per_page,
            history.start_date.isoformat(),
            history.end_date.isoformat(),
            dashboard_enabled=dashboard_enabled,
        )
        return render(request, "sentiment_app/twitter.html", context)

    if request.GET.get("show") != "1":
        request.session.pop(TWITTER_RESULT_SESSION_KEY, None)
        return render(request, "sentiment_app/twitter.html", context)

    saved_result = request.session.get(TWITTER_RESULT_SESSION_KEY) or {}
    saved_count = _safe_positive_int(saved_result.get("tweet_count"), 0)
    last_page = _safe_positive_int(saved_result.get("last_page"), 0)
    last_per_page = _normalize_per_page(saved_result.get("last_per_page"), 0)
    saved_history_id = _safe_positive_int(saved_result.get("history_id"), 0)

    if not saved_history_id:
        request.session.pop(TWITTER_RESULT_SESSION_KEY, None)
        return render(request, "sentiment_app/twitter.html", context)

    saved_history = ScrapeHistory.objects.filter(id=saved_history_id, user=request.user).only(
        "rows",
        "tweet_count",
        "start_date",
        "end_date",
    ).first()
    if saved_history is None:
        request.session.pop(TWITTER_RESULT_SESSION_KEY, None)
        return render(request, "sentiment_app/twitter.html", context)

    saved_rows = _load_scrape_rows(saved_history)
    if not saved_rows:
        request.session.pop(TWITTER_RESULT_SESSION_KEY, None)
        return render(request, "sentiment_app/twitter.html", context)

    requested_page = _safe_positive_int(request.GET.get("page"), 1)
    per_page = _normalize_per_page(request.GET.get("per_page"), DEFAULT_PER_PAGE)
    if last_page and requested_page == last_page and per_page == last_per_page:
        messages.warning(
            request,
            "Hasil scraping bersifat sementara dan dibersihkan setelah refresh.",
        )
        request.session.pop(TWITTER_RESULT_SESSION_KEY, None)
        return render(request, "sentiment_app/twitter.html", context)

    if not _apply_scraping_context(
        context,
        saved_rows,
        saved_count or int(saved_history.tweet_count or 0),
        requested_page,
        per_page,
        saved_history.start_date.isoformat(),
        saved_history.end_date.isoformat(),
        dashboard_enabled=dashboard_enabled,
    ):
        request.session.pop(TWITTER_RESULT_SESSION_KEY, None)
        return render(request, "sentiment_app/twitter.html", context)

    saved_result["last_page"] = int(context.get("current_page", 1))
    saved_result["last_per_page"] = per_page
    request.session[TWITTER_RESULT_SESSION_KEY] = saved_result
    request.session.modified = True

    return render(request, "sentiment_app/twitter.html", context)


@login_required
def download_output_view(request: HttpRequest, filename: str) -> FileResponse:
    if not SAFE_OUTPUT_RE.match(filename):
        raise Http404("Nama file tidak valid.")

    outputs_dir = (Path(settings.MEDIA_ROOT) / "outputs").resolve()
    target_file = (outputs_dir / filename).resolve()

    if outputs_dir not in target_file.parents:
        raise Http404("Path tidak valid.")
    if not target_file.exists() or not target_file.is_file():
        raise Http404("File tidak ditemukan.")

    return FileResponse(target_file.open("rb"), as_attachment=True, filename=filename)
