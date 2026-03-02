from __future__ import annotations

import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

import requests

BASE_URL = "https://api.twitterapi.io"
SEARCH_ENDPOINT = "/twitter/tweet/advanced_search"
REQUEST_TIMEOUT_SECONDS = (5, 20)
WINDOW_DAYS = 1
MAX_TWEETS_PER_WINDOW = 500
MAX_TOTAL_TWEETS = 4000
PAGE_SLEEP_SECONDS = 0.15
WINDOW_SLEEP_SECONDS = 0.1


class TwitterAPIError(RuntimeError):
    pass


def _clean_query(query: str) -> str:
    cleaned = query.strip()
    cleaned = re.sub(r"(?i)\b(?:since|until):\S+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _coalesce(tweet: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = tweet.get(key)
        if value not in (None, ""):
            return value
    return ""


def _extract_tweets(data: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("tweets", "data", "results"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    return []


def _to_author(tweet: dict[str, Any]) -> str:
    author = tweet.get("author") or tweet.get("user") or {}
    if isinstance(author, dict):
        return (
            str(
                author.get("userName")
                or author.get("username")
                or author.get("screen_name")
                or author.get("name")
                or ""
            )
            .strip()
        )
    return str(tweet.get("userName") or tweet.get("username") or "").strip()


def _to_username(tweet: dict[str, Any], author: str) -> str:
    if author:
        return author
    user_name = str(tweet.get("userName") or tweet.get("username") or "").strip()
    return user_name


def _to_url(tweet: dict[str, Any], author: str) -> str:
    direct_url = tweet.get("url") or tweet.get("permalink")
    if direct_url:
        return str(direct_url)
    tweet_id = str(tweet.get("id") or tweet.get("tweet_id") or "").strip()
    if tweet_id and author:
        return f"https://x.com/{author}/status/{tweet_id}"
    return ""


def _to_image_url(tweet: dict[str, Any]) -> str:
    image = tweet.get("image_tweet")
    if image:
        return str(image)

    photos = tweet.get("photos")
    if isinstance(photos, list):
        for item in photos:
            if isinstance(item, dict):
                image_url = item.get("url") or item.get("src")
                if image_url:
                    return str(image_url)

    media = tweet.get("media")
    if isinstance(media, list):
        for item in media:
            if isinstance(item, dict):
                image_url = item.get("media_url_https") or item.get("media_url") or item.get("url")
                if image_url:
                    return str(image_url)

    extended_entities = tweet.get("extendedEntities") or tweet.get("extended_entities") or {}
    if isinstance(extended_entities, dict):
        media_items = extended_entities.get("media")
        if isinstance(media_items, list):
            for media in media_items:
                media_url = media.get("media_url_https") or media.get("media_url")
                if media_url:
                    return str(media_url)
    if isinstance(extended_entities, list):
        for media in extended_entities:
            if isinstance(media, dict):
                media_url = media.get("media_url_https") or media.get("media_url")
                if media_url:
                    return str(media_url)
    if isinstance(extended_entities, str):
        marker = "media_url_https"
        if marker in extended_entities:
            parts = extended_entities.split(marker, 1)[-1]
            quote_split = parts.split("'")
            if len(quote_split) >= 3:
                return quote_split[2]
    return ""


def normalize_tweets(
    raw_tweets: list[dict[str, Any]],
    week_start: str,
    week_end: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tweet in raw_tweets:
        text = str(tweet.get("text") or tweet.get("full_text") or "").strip()
        if not text:
            continue

        author = _to_author(tweet)
        username = _to_username(tweet, author)
        created_at = str(_coalesce(tweet, "CreatedAt", "createdAt", "created_at", "date")).strip()
        retweet_count = _coalesce(tweet, "retweetCount", "retweet_count")
        reply_count = _coalesce(tweet, "replyCount", "reply_count")
        like_count = _coalesce(tweet, "likeCount", "favorite_count", "like_count")
        quote_count = _coalesce(tweet, "quoteCount", "quote_count")
        view_count = _coalesce(tweet, "viewCount", "view_count")
        lang = str(_coalesce(tweet, "lang", "language")).strip()
        bookmark_count = _coalesce(tweet, "bookmarkCount", "bookmark_count")
        in_reply_to = str(_coalesce(tweet, "inReplyTold", "inReplyToId", "in_reply_to_status_id_str")).strip()
        is_reply = _coalesce(tweet, "isReply", "is_reply")
        if is_reply == "":
            is_reply = bool(in_reply_to)

        normalized.append(
            {
                "id": str(_coalesce(tweet, "id", "tweet_id", "tweetId", "rest_id")).strip(),
                "url": _to_url(tweet, username),
                "text": text,
                "retweetCount": retweet_count,
                "replyCount": reply_count,
                "likeCount": like_count,
                "quoteCount": quote_count,
                "viewCount": view_count,
                "CreatedAt": created_at,
                "lang": lang,
                "bookmarkCount": bookmark_count,
                "isReply": is_reply,
                "inReplyTold": in_reply_to,
                "userName": username,
                "image_tweet": _to_image_url(tweet),
                # metadata internal untuk jejak per-minggu, tidak ikut diekspor default.
                "_week_start": week_start,
                "_week_end": week_end,
            }
        )
    return normalized


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise TwitterAPIError("Format tanggal harus YYYY-MM-DD.") from exc


def _parse_created_at_date(value: Any) -> date | None:
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


def _fetch_window_tweets(
    api_key: str,
    query: str,
    max_tweets_per_window: int = MAX_TWEETS_PER_WINDOW,
) -> list[dict[str, Any]]:
    url = f"{BASE_URL}{SEARCH_ENDPOINT}"
    headers = {
        "X-API-Key": api_key,
        "Accept": "application/json",
    }

    cursor = ""
    fetched: list[dict[str, Any]] = []
    tweet_count = 0
    while tweet_count < max_tweets_per_window:
        params: dict[str, Any] = {
            "query": query,
            "queryType": "Latest",
        }
        if cursor:
            params["cursor"] = cursor

        try:
            response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            raise TwitterAPIError(f"Permintaan ke Twitter API gagal: {exc}") from exc

        if response.status_code in (401, 403):
            raise TwitterAPIError("API key tidak valid atau akses tidak diizinkan.")
        if response.status_code == 429:
            break
        if response.status_code >= 400:
            break

        try:
            payload = response.json()
        except ValueError as exc:
            raise TwitterAPIError("Twitter API mengembalikan respons non-JSON.") from exc

        raw_tweets = _extract_tweets(payload)
        if not raw_tweets:
            break

        for tweet in raw_tweets:
            if tweet_count >= max_tweets_per_window:
                break
            fetched.append(tweet)
            tweet_count += 1

        has_next_page = bool(payload.get("has_next_page"))
        next_cursor = payload.get("next_cursor")
        if not has_next_page or not next_cursor:
            break
        cursor = str(next_cursor)
        time.sleep(PAGE_SLEEP_SECONDS)

    return fetched


def fetch_tweets(
    api_key: str,
    query: str | None = None,
    language: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_tweets_per_window: int = MAX_TWEETS_PER_WINDOW,
    max_total_tweets: int = MAX_TOTAL_TWEETS,
    max_range_days: int | None = None,
    window_days: int = WINDOW_DAYS,
    on_window: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    if not api_key:
        raise TwitterAPIError("API key wajib diisi.")
    if not query:
        raise TwitterAPIError("Isi kueri.")

    today = date.today()
    window_days = max(1, int(window_days))
    parsed_end = _parse_date(end_date) if end_date else today
    parsed_start = _parse_date(start_date) if start_date else (parsed_end - timedelta(days=window_days))
    parsed_end_exclusive = parsed_end + timedelta(days=1)

    if parsed_start >= parsed_end_exclusive:
        raise TwitterAPIError("Tanggal mulai harus lebih kecil dari tanggal selesai.")

    max_tweets_per_window = max(1, int(max_tweets_per_window))
    max_total_tweets = max(1, int(max_total_tweets))
    if max_range_days is not None:
        max_range_days = max(1, int(max_range_days))
        selected_days = (parsed_end - parsed_start).days + 1
        if selected_days > max_range_days:
            raise TwitterAPIError(
                f"Rentang scraping terlalu panjang ({selected_days} hari). "
                f"Maksimal {max_range_days} hari per permintaan."
            )

    base_query = _clean_query(query)
    if not base_query:
        raise TwitterAPIError("Kueri tidak valid setelah menghapus operator tanggal. Isi kata kunci utama tanpa since/until.")

    if language:
        base_query = f"{base_query} lang:{language}"

    all_tweets: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    total_raw_tweets = 0
    total_out_of_range = 0
    total_unparseable_date = 0
    kept_total = 0
    window_cursor = parsed_start
    while window_cursor < parsed_end_exclusive:
        next_window = min(window_cursor + timedelta(days=window_days), parsed_end_exclusive)
        since_str = window_cursor.strftime("%Y-%m-%d")
        until_str = next_window.strftime("%Y-%m-%d")
        window_query = f"{base_query} since:{since_str} until:{until_str}"

        raw_window_tweets = _fetch_window_tweets(
            api_key=api_key,
            query=window_query,
            max_tweets_per_window=max_tweets_per_window,
        )
        total_raw_tweets += len(raw_window_tweets)
        normalized_window_tweets = normalize_tweets(raw_window_tweets, week_start=since_str, week_end=until_str)
        kept_in_window: list[dict[str, Any]] = []

        for tweet in normalized_window_tweets:
            if kept_total >= max_total_tweets:
                break
            created_date = _parse_created_at_date(tweet.get("CreatedAt"))
            if created_date is None:
                total_unparseable_date += 1
            else:
                # Keep rows by global selected range. Per-window strict filter can
                # drop valid rows when API date boundaries/timezones are inconsistent.
                if not (parsed_start <= created_date <= parsed_end):
                    total_out_of_range += 1
                    continue

            if not tweet.get("_week_start"):
                tweet["_week_start"] = since_str
            if not tweet.get("_week_end"):
                tweet["_week_end"] = until_str

            tweet_id = str(tweet.get("id", "")).strip()
            if tweet_id:
                dedup_key = f"id:{tweet_id}"
            else:
                dedup_user = str(tweet.get("userName", "")).strip().lower()
                dedup_created = str(tweet.get("CreatedAt", "")).strip()
                dedup_text = str(tweet.get("text", "")).strip().lower()
                dedup_key = f"fallback:{dedup_user}|{dedup_created}|{dedup_text[:180]}"

            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            kept_in_window.append(tweet)
            kept_total += 1

        if kept_in_window:
            if on_window is not None:
                on_window(kept_in_window)
            else:
                all_tweets.extend(kept_in_window)

        if kept_total >= max_total_tweets:
            break

        window_cursor = next_window
        time.sleep(WINDOW_SLEEP_SECONDS)

    if kept_total == 0 and total_raw_tweets > 0:
        raise TwitterAPIError(
            "API mengembalikan data di luar rentang tanggal yang dipilih, "
            "atau format tanggal tweet tidak bisa dibaca. "
            f"(Raw: {total_raw_tweets}, di luar rentang: {total_out_of_range}, tanggal tidak terbaca: {total_unparseable_date}). "
            "Coba rentang tanggal yang lebih baru, dan cek dukungan histori pada plan twitterapi.io Anda."
        )

    if on_window is not None:
        return []
    return all_tweets
