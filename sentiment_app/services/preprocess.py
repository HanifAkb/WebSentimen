from __future__ import annotations

from .model_service import (
    EMOJI_RE,
    HASHTAG_RE,
    NON_ALPHA_RE,
    NUMBER_RE,
    URL_RE,
    USER_RE,
    WS_RE,
    _get_stemmer,
    _load_slang_map,
    _load_stopwords,
    _normalize_slang,
    _remove_stopwords,
    _stem_text,
    preprocess_text,
)

__all__ = [
    "EMOJI_RE",
    "HASHTAG_RE",
    "NON_ALPHA_RE",
    "NUMBER_RE",
    "URL_RE",
    "USER_RE",
    "WS_RE",
    "_get_stemmer",
    "_load_slang_map",
    "_load_stopwords",
    "_normalize_slang",
    "_remove_stopwords",
    "_stem_text",
    "preprocess_text",
]
