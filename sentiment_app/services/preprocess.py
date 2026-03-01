from __future__ import annotations

import re
from pathlib import Path

from django.conf import settings

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
USER_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
NUMBER_RE = re.compile(r"\d+")
NON_ALPHA_RE = re.compile(r"[^a-z\s']+")
WS_RE = re.compile(r"\s+")
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002500-\U00002BEF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F926-\U0001F937"
    "\U00010000-\U0010FFFF"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"
    "\u3030"
    "]+",
    flags=re.UNICODE,
)

DEFAULT_STOPWORDS = {
    "yang",
    "dan",
    "di",
    "ke",
    "dari",
    "untuk",
    "dengan",
    "itu",
    "ini",
    "atau",
    "karena",
    "pada",
    "adalah",
    "juga",
    "sudah",
    "saja",
    "lagi",
    "saya",
    "kamu",
    "dia",
    "kami",
    "kita",
    "mereka",
}

SLANG_MAP = {
    "ajg": "anjing",
    "anj": "anjing",
    "btw": "by the way",
    "biasaaja": "biasa saja",
    "bgtu": "begitu",
    "blm": "belum",
    "bener": "benar",
    "bgt": "banget",
    "brarti": "berarti",
    "brrti": "berarti",
    "bnyk": "banyak",
    "bbrp": "beberapa",
    "brp": "berapa",
    "bkn": "bukan",
    "bs": "bisa",
    "kampret": "kasar",
    "tai": "kasar",
    "bgst": "bangsat",
    "cuan": "untung",
    "dgn": "dengan",
    "dg": "dengan",
    "dimana": "di mana",
    "dlm": "dalam",
    "dll": "dan lain lain",
    "deh": "dah",
    "dpt": "dapat",
    "dr": "dari",
    "dlu": "dahulu",
    "dl": "dahulu",
    "dy": "dia",
    "gimana": "bagaimana",
    "gmna": "bagaimana",
    "gmn": "bagaimana",
    "gtw": "tidak tau",
    "gpp": "tidak apa apa",
    "jg": "juga",
    "gue": "saya",
    "gua": "saya",
    "gw": "saya",
    "sy": "saya",
    "elo": "kamu",
    "lu": "kamu",
    "loe": "kamu",
    "lo": "kamu",
    "kt": "kata",
    "ktnya": "katanya",
    "krn": "karena",
    "karna": "karena",
    "klau": "kalau",
    "klo": "kalau",
    "klu": "kalau",
    "kalo": "kalau",
    "kek": "kayak",
    "kyk": "kayak",
    "knp": "kenapa",
    "kpd": "kepada",
    "napa": "kenapa",
    "npa": "kenapa",
    "ngab": "abang",
    "kpn": "kapan",
    "kpan": "kapan",
    "kmrn": "kemarin",
    "maren": "kemarin",
    "lgsg": "langsung",
    "lg": "lagi",
    "mls": "malas",
    "mantul": "mantap sekali",
    "msh": "masih",
    "mnrt": "menurut",
    "mhn": "mohon",
    "ntr": "nanti",
    "ntar": "nanti",
    "sdg": "sedang",
    "sdng": "sedang",
    "smkn": "semakin",
    "otw": "sedang dalam perjalanan",
    "sm": "sama",
    "smpe": "sampai",
    "sampe": "sampai",
    "ampe": "sampai",
    "tmn": "teman",
    "tp": "tapi",
    "tgl": "tanggal",
    "trs": "terus",
    "trus": "terus",
    "tsb": "tersebut",
    "ttg": "tentang",
    "tgs": "tugas",
    "sbnrnya": "sebenarnya",
    "sbg": "sebagai",
    "sbgi": "sebagai",
    "skrg": "sekarang",
    "spt": "seperti",
    "sdt": "sedikit",
    "thn": "tahun",
    "udh": "sudah",
    "udah": "sudah",
    "sdh": "sudah",
    "utk": "untuk",
    "yg": "yang",
    "nyg": "yang",
    "neh": "nih",
    "ygy": "ya guys ya",
    "ny": "nya",
    "wkwkwk": "tertawa",
    "wkt": "waktu",
    "lol": "tertawa",
    "wk": "tertawa",
    "gt": "gitu",
    "aj": "saja",
    "aja": "saja",
    "pd": "pada",
    "pake": "pakai",
    "hr": "hari",
    "jd": "jadi",
    "mjd": "jadi",
    "org": "orang",
    "td": "tadi",
    "tdk": "tidak",
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "panikga": "panik tidak",
    "haha": "tertawa",
    "hehe": "tertawa",
}

_STOPWORDS_CACHE: set[str] | None = None
_STEMMER_CACHE = None
_STEMMER_ATTEMPTED = False


def _load_stopwords() -> set[str]:
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is not None:
        return _STOPWORDS_CACHE

    candidates = [
        Path(settings.BASE_DIR) / "stopwords-id.txt",
        Path(settings.BASE_DIR) / "sentiment_site" / "models" / "stopwords-id.txt",
    ]

    for path in candidates:
        if path.exists():
            try:
                loaded = {
                    line.strip().lower()
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                }
                if loaded:
                    _STOPWORDS_CACHE = loaded
                    return _STOPWORDS_CACHE
            except Exception:
                pass

    _STOPWORDS_CACHE = set(DEFAULT_STOPWORDS)
    return _STOPWORDS_CACHE


def _normalize_slang(text: str) -> str:
    tokens = text.split()
    replaced = [SLANG_MAP.get(token, token) for token in tokens]
    return " ".join(replaced)


def _remove_stopwords(text: str) -> str:
    stopwords = _load_stopwords()
    return " ".join(word for word in text.split() if word not in stopwords)


def _get_stemmer():
    global _STEMMER_CACHE, _STEMMER_ATTEMPTED
    if _STEMMER_ATTEMPTED:
        return _STEMMER_CACHE

    _STEMMER_ATTEMPTED = True
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

        _STEMMER_CACHE = StemmerFactory().create_stemmer()
    except Exception:
        _STEMMER_CACHE = None
    return _STEMMER_CACHE


def _stem_text(text: str) -> str:
    stemmer = _get_stemmer()
    if stemmer is None:
        return text

    tokens = text.split()
    stemmed_words = [stemmer.stem(word) for word in tokens]
    return " ".join(stemmed_words)


def preprocess_text(text: str) -> str:
    normalized = str(text or "").lower()
    normalized = EMOJI_RE.sub(" ", normalized)
    normalized = URL_RE.sub(" ", normalized)
    normalized = USER_RE.sub(" ", normalized)
    normalized = NUMBER_RE.sub(" ", normalized)
    normalized = HASHTAG_RE.sub(" ", normalized)
    normalized = NON_ALPHA_RE.sub(" ", normalized)
    normalized = WS_RE.sub(" ", normalized).strip()

    normalized = _normalize_slang(normalized)
    normalized = _remove_stopwords(normalized)
    normalized = _stem_text(normalized)
    normalized = WS_RE.sub(" ", normalized).strip()
    return normalized
