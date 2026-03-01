from __future__ import annotations

import base64
import io
import re
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

from django.conf import settings
from django.contrib.auth import login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import FileResponse, Http404, HttpRequest, HttpResponse, HttpResponseForbidden
from django.shortcuts import get_object_or_404
from django.shortcuts import redirect, render
from django.urls import reverse

from .forms import AdminCreateUserForm, LoginForm, PredictForm, TwitterFetchForm
from .models import ScrapeHistory
from .services.file_service import (
    FileValidationError,
    generate_classification_csv,
    parse_uploaded_file,
)
from .services.model_service import ModelServiceError, predict_batch, predict_single
from .services.twitter_client import TwitterAPIError, fetch_tweets

try:
    from wordcloud import STOPWORDS as WORDCLOUD_BASE_STOPWORDS
    from wordcloud import WordCloud
except Exception:
    WORDCLOUD_BASE_STOPWORDS = set()
    WordCloud = None

SAFE_OUTPUT_RE = re.compile(r"^[A-Za-z0-9_-]+\.csv$")
DEFAULT_PER_PAGE = 10
MAX_PER_PAGE = 200
TWITTER_RESULT_SESSION_KEY = "twitter_last_result"
PREDICTION_COLUMNS = ["knn_label", "knn_score", "svm_label", "svm_score"]
PREDICTION_HEADERS = {
    "knn_label": "KNN",
    "knn_score": "Skor KNN",
    "svm_label": "SVM",
    "svm_score": "Skor SVM",
}
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


def _safe_positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _normalize_per_page(value: object, default: int = DEFAULT_PER_PAGE) -> int:
    parsed = _safe_positive_int(value, default)
    return min(parsed, MAX_PER_PAGE)


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

    raw = str(value).strip()
    if not raw:
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


def _bucket_start(value: date, granularity: str) -> date:
    if granularity == "week":
        return value - timedelta(days=value.weekday())
    if granularity == "month":
        return value.replace(day=1)
    return value


def _next_bucket(value: date, granularity: str) -> date:
    if granularity == "day":
        return value + timedelta(days=1)
    if granularity == "week":
        return value + timedelta(days=7)
    year = value.year + (1 if value.month == 12 else 0)
    month = 1 if value.month == 12 else value.month + 1
    return date(year, month, 1)


def _format_bucket_label(value: date, granularity: str) -> str:
    if granularity == "day":
        return value.strftime("%d %b %Y")
    if granularity == "week":
        week_end = value + timedelta(days=6)
        return f"{value.strftime('%d %b')} - {week_end.strftime('%d %b %Y')}"
    return value.strftime("%b %Y")


def _normalize_sentiment_label(value: object) -> str:
    text = str(value or "").strip().lower()
    if text == "positive":
        return "Positive"
    if text == "negative":
        return "Negative"
    return "Other"


def _clean_text_for_wordcloud(text: str) -> str:
    cleaned = text.lower()
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
    cleaned = re.sub(r"[@#]\w+", " ", cleaned)
    cleaned = re.sub(r"[^0-9a-zA-Z_\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _build_wordcloud_image(texts: list[str], colormap: str) -> str | None:
    if WordCloud is None:
        return None

    cleaned_texts = [_clean_text_for_wordcloud(str(text or "")) for text in texts if str(text or "").strip()]
    combined_text = " ".join(text for text in cleaned_texts if text)
    if not combined_text:
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
        prefer_horizontal=0.95,
        colormap=colormap,
    ).generate(combined_text)

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
) -> dict[str, object]:
    knn_positive_texts: list[str] = []
    knn_negative_texts: list[str] = []
    svm_positive_texts: list[str] = []
    svm_negative_texts: list[str] = []
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
                knn_positive_texts.append(text)
            elif knn_label == "Negative":
                knn_negative_texts.append(text)

            if svm_label == "Positive":
                svm_positive_texts.append(text)
            elif svm_label == "Negative":
                svm_negative_texts.append(text)

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
    trend_counter: Counter[date] = Counter()
    for created_date in row_dates:
        trend_counter[_bucket_start(created_date, granularity)] += 1

    chart_labels: list[str] = []
    chart_values: list[int] = []
    bucket_cursor = _bucket_start(start_date, granularity)
    bucket_last = _bucket_start(end_date, granularity)
    while bucket_cursor <= bucket_last:
        chart_labels.append(_format_bucket_label(bucket_cursor, granularity))
        chart_values.append(int(trend_counter.get(bucket_cursor, 0)))
        bucket_cursor = _next_bucket(bucket_cursor, granularity)

    wordcloud_error = ""
    wordclouds: dict[str, str | None] = {
        "knn_positive_image": None,
        "knn_negative_image": None,
        "svm_positive_image": None,
        "svm_negative_image": None,
    }
    if WordCloud is None:
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
            "other": int(knn_counts.get("Other", 0)),
        },
        "svm_counts": {
            "positive": int(svm_counts.get("Positive", 0)),
            "negative": int(svm_counts.get("Negative", 0)),
            "other": int(svm_counts.get("Other", 0)),
        },
        "wordcloud_available": WordCloud is not None and not wordcloud_error,
        "wordcloud_error": wordcloud_error,
        "wordclouds": wordclouds,
        "charts": {
            "pie_labels": ["Positif", "Negatif", "Lainnya"],
            "knn_pie": [
                int(knn_counts.get("Positive", 0)),
                int(knn_counts.get("Negative", 0)),
                int(knn_counts.get("Other", 0)),
            ],
            "svm_pie": [
                int(svm_counts.get("Positive", 0)),
                int(svm_counts.get("Negative", 0)),
                int(svm_counts.get("Other", 0)),
            ],
            "trend_labels": chart_labels,
            "trend_values": chart_values,
            "trend_title": f"Jumlah Tweet per {granularity_label}",
        },
    }


def _is_admin_user(user) -> bool:
    return bool(getattr(user, "is_authenticated", False) and getattr(user, "is_superuser", False))


def _apply_scraping_context(
    context: dict[str, object],
    rows: list[dict[str, object]],
    tweet_count: int,
    requested_page: int,
    per_page: int,
    start_date_value: object,
    end_date_value: object,
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
    context["dashboard"] = _build_scraping_dashboard(
        rows,
        _safe_parse_iso_date(start_date_value),
        _safe_parse_iso_date(end_date_value),
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


def _serialize_history_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    for row in rows:
        serialized.append({str(key): _json_safe_value(value) for key, value in row.items()})
    return serialized


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
def register_user_view(request: HttpRequest) -> HttpResponse:
    if not _is_admin_user(request.user):
        return HttpResponseForbidden("Hanya admin yang boleh membuat akun baru.")

    form = AdminCreateUserForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        new_user = form.save()
        messages.success(request, f"Akun baru berhasil dibuat: {new_user.username}")
        return redirect("register_user")
    return render(request, "sentiment_app/register.html", {"form": form})


@login_required
def history_list_view(request: HttpRequest) -> HttpResponse:
    histories = ScrapeHistory.objects.filter(user=request.user).only(
        "id",
        "query",
        "language",
        "start_date",
        "end_date",
        "tweet_count",
        "created_at",
    )
    context = {
        "histories": histories,
    }
    return render(request, "sentiment_app/history.html", context)


@login_required
def history_detail_view(request: HttpRequest, history_id: int) -> HttpResponse:
    history = get_object_or_404(ScrapeHistory, id=history_id, user=request.user)
    form = TwitterFetchForm()
    requested_page = _safe_positive_int(request.GET.get("page"), 1)
    per_page = _normalize_per_page(request.GET.get("per_page"), DEFAULT_PER_PAGE)
    rows = history.rows if isinstance(history.rows, list) else []

    context: dict[str, object] = {
        "form": form,
        "history_mode": True,
        "history": history,
    }
    if not _apply_scraping_context(
        context,
        rows,
        int(history.tweet_count or 0),
        requested_page,
        per_page,
        history.start_date.isoformat(),
        history.end_date.isoformat(),
    ):
        messages.warning(request, "Data riwayat scraping kosong.")

    return render(request, "sentiment_app/twitter.html", context)


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
            elif upload_file:
                texts, detected_column, source_rows, source_columns = parse_uploaded_file(upload_file, text_column)
                predictions = predict_batch(texts)
                output_filename = generate_classification_csv(predictions, prefix="uploaded")
                preview_headers, preview_rows = _build_batch_preview(source_rows, source_columns, predictions)
                context["batch_count"] = len(predictions)
                context["detected_column"] = detected_column
                context["batch_preview_headers"] = preview_headers
                context["batch_preview_rows"] = preview_rows[:20]
                context["output_filename"] = output_filename
                context["download_url"] = reverse("download_output", args=[output_filename])
                context["active_tab"] = "file"
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

        if not api_key:
            messages.error(request, "API key wajib diisi.")
            return redirect("twitter_fetch")

        try:
            tweets = fetch_tweets(
                api_key=api_key,
                query=query,
                language=language,
                start_date=start_date.isoformat() if start_date else None,
                end_date=end_date.isoformat() if end_date else None,
            )
            if not tweets:
                messages.warning(request, "Tidak ada tweet yang ditemukan untuk permintaan ini.")
                request.session.pop(TWITTER_RESULT_SESSION_KEY, None)
                return redirect("twitter_fetch")

            predictions = predict_batch([tweet["text"] for tweet in tweets])

            classified_rows = []
            for tweet, prediction in zip(tweets, predictions):
                classified_rows.append(
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

            # Guard terakhir di layer view: tampilkan hanya tanggal dalam rentang user.
            filtered_rows: list[dict[str, object]] = []
            for row in classified_rows:
                created_date = _parse_created_at_date(row.get("CreatedAt"))
                if created_date is None:
                    continue
                if start_date and created_date < start_date:
                    continue
                if end_date and created_date > end_date:
                    continue
                filtered_rows.append(row)

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
            )

            request.session[TWITTER_RESULT_SESSION_KEY] = {
                "rows": history_rows,
                "tweet_count": len(history_rows),
                "start_date": start_date.isoformat() if start_date else "",
                "end_date": end_date.isoformat() if end_date else "",
                "history_id": history_item.id,
                "last_page": 0,
                "last_per_page": 0,
            }
            request.session.modified = True

            query_url = reverse("twitter_fetch")
            return redirect(f"{query_url}?show=1&page=1&per_page={per_page}")
        except (TwitterAPIError, ModelServiceError, FileValidationError) as exc:
            messages.error(request, str(exc))
            return redirect("twitter_fetch")
        except Exception as exc:
            messages.error(request, f"Terjadi kesalahan tak terduga saat mengambil/mengklasifikasikan tweet: {exc}")
            return redirect("twitter_fetch")

    form = TwitterFetchForm()
    context: dict[str, object] = {
        "form": form,
    }

    history_id = _safe_positive_int(request.GET.get("history"), 0)
    if history_id:
        history = get_object_or_404(ScrapeHistory, id=history_id, user=request.user)
        context["history_mode"] = True
        context["history"] = history
        requested_page = _safe_positive_int(request.GET.get("page"), 1)
        per_page = _normalize_per_page(request.GET.get("per_page"), DEFAULT_PER_PAGE)
        rows = history.rows if isinstance(history.rows, list) else []
        _apply_scraping_context(
            context,
            rows,
            int(history.tweet_count or 0),
            requested_page,
            per_page,
            history.start_date.isoformat(),
            history.end_date.isoformat(),
        )
        return render(request, "sentiment_app/twitter.html", context)

    if request.GET.get("show") != "1":
        request.session.pop(TWITTER_RESULT_SESSION_KEY, None)
        return render(request, "sentiment_app/twitter.html", context)

    saved_result = request.session.get(TWITTER_RESULT_SESSION_KEY) or {}
    saved_rows = saved_result.get("rows")
    saved_count = _safe_positive_int(saved_result.get("tweet_count"), 0)
    last_page = _safe_positive_int(saved_result.get("last_page"), 0)
    last_per_page = _normalize_per_page(saved_result.get("last_per_page"), 0)

    if not isinstance(saved_rows, list) or not saved_rows:
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
        saved_count,
        requested_page,
        per_page,
        saved_result.get("start_date"),
        saved_result.get("end_date"),
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
