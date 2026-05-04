from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from django.conf import settings
from sklearn.exceptions import NotFittedError

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

DEFAULT_SLANG_MAP = {
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
SLANG_MAP = DEFAULT_SLANG_MAP

_STOPWORDS_CACHE: set[str] | None = None
_SLANG_MAP_CACHE: dict[str, str] | None = None
_STEMMER_CACHE = None
_STEMMER_ATTEMPTED = False


class ModelServiceError(RuntimeError):
    pass


@dataclass
class ModelArtifacts:
    knn_model: Any
    svm_model: Any
    vectorizer: Any | None = None
    label_encoder: Any | None = None


_ARTIFACTS_CACHE: ModelArtifacts | None = None
KNN_NEUTRAL_MIN = 0.45
KNN_NEUTRAL_MAX = 0.55
SVM_NEUTRAL_MIN = 0.45
SVM_NEUTRAL_MAX = 0.55


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


def _load_slang_map() -> dict[str, str]:
    global _SLANG_MAP_CACHE
    if _SLANG_MAP_CACHE is not None:
        return _SLANG_MAP_CACHE

    candidates = [
        _models_dir() / "singkatan.tsv",
        Path(settings.BASE_DIR) / "singkatan.tsv",
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            loaded: dict[str, str] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split("\t", 1)
                if len(parts) != 2:
                    continue
                source = parts[0].strip().lower()
                replacement = parts[1].strip().lower()
                if source and replacement:
                    loaded[source] = replacement
            if loaded:
                _SLANG_MAP_CACHE = loaded
                return _SLANG_MAP_CACHE
        except Exception:
            pass

    _SLANG_MAP_CACHE = dict(DEFAULT_SLANG_MAP)
    return _SLANG_MAP_CACHE


def _normalize_slang(text: str) -> str:
    slang_map = _load_slang_map()
    tokens = text.split()
    replaced = [slang_map.get(token, token) for token in tokens]
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


def preprocess_text(text: str, apply_stemming: bool = True) -> str:
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
    if apply_stemming:
        normalized = _stem_text(normalized)
    normalized = WS_RE.sub(" ", normalized).strip()
    return normalized


def clear_cache() -> None:
    global _ARTIFACTS_CACHE
    _ARTIFACTS_CACHE = None


def _models_dir() -> Path:
    return Path(getattr(settings, "SENTIMENT_MODELS_DIR", settings.BASE_DIR / "sentiment_site" / "models"))


def _find_artifact_path(filename: str) -> Path | None:
    primary = _models_dir() / filename
    fallback = Path(settings.BASE_DIR) / filename
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return None


def _load_artifacts() -> ModelArtifacts:
    global _ARTIFACTS_CACHE
    if _ARTIFACTS_CACHE is not None:
        return _ARTIFACTS_CACHE

    models_dir = _models_dir()
    knn_path = _find_artifact_path("knn_model.joblib")
    svm_candidates = ["svm_linear_model.joblib", "svm_rbf_model.joblib"]
    svm_path = next((path for path in (_find_artifact_path(name) for name in svm_candidates) if path is not None), None)

    if knn_path is None or svm_path is None:
        missing = []
        if knn_path is None:
            missing.append("knn_model.joblib")
        if svm_path is None:
            missing.append(" / ".join(svm_candidates))
        raise ModelServiceError(
            "File model wajib tidak ditemukan: "
            f"{', '.join(missing)}. Letakkan file tersebut di {models_dir}."
        )

    try:
        knn_model = joblib.load(knn_path)
        svm_model = joblib.load(svm_path)
    except Exception as exc:
        raise ModelServiceError(f"Gagal memuat file model: {exc}") from exc

    vectorizer = None
    for vectorizer_name in ("vectorizer.joblib", "tfidf_vectorizer.joblib"):
        vectorizer_path = _find_artifact_path(vectorizer_name)
        if vectorizer_path is not None:
            try:
                vectorizer = joblib.load(vectorizer_path)
            except Exception as exc:
                raise ModelServiceError(f"Gagal memuat {vectorizer_name}: {exc}") from exc
            break

    label_encoder = None
    label_encoder_path = _find_artifact_path("label_encoder.joblib")
    if label_encoder_path is not None:
        try:
            label_encoder = joblib.load(label_encoder_path)
        except Exception as exc:
            raise ModelServiceError(f"Gagal memuat label_encoder.joblib: {exc}") from exc

    _ARTIFACTS_CACHE = ModelArtifacts(
        knn_model=knn_model,
        svm_model=svm_model,
        vectorizer=vectorizer,
        label_encoder=label_encoder,
    )
    return _ARTIFACTS_CACHE


def _row_count(features: Any) -> int:
    if hasattr(features, "shape"):
        return int(features.shape[0])
    return len(features)


def _positive_class_index(classes: Any) -> int | None:
    if classes is None:
        return None

    for index, cls in enumerate(classes):
        value = str(cls).strip().lower()
        if value in {"1", "positive", "pos", "true"} or "posit" in value:
            return index
    return None


def _score_from_proba(probabilities: Any, classes: Any) -> list[float]:
    values = np.asarray(probabilities)
    if values.ndim == 1:
        return [float(score) for score in values]

    positive_index = _positive_class_index(classes)
    if positive_index is None:
        positive_index = 1 if values.shape[1] > 1 else 0

    return [float(row[positive_index]) for row in values]


def _sigmoid(values: Any) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _score_from_decision(decision_values: Any, classes: Any) -> list[float]:
    values = np.asarray(decision_values)
    if values.ndim == 1:
        selected = values
    else:
        positive_index = _positive_class_index(classes)
        if positive_index is None:
            positive_index = 1 if values.shape[1] > 1 else 0
        selected = values[:, positive_index]

    probabilities = _sigmoid(selected)
    return [float(value) for value in probabilities]


def _extract_scores(model: Any, features: Any) -> list[float | None]:
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(features)
            return _score_from_proba(probabilities, getattr(model, "classes_", None))
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        try:
            decisions = model.decision_function(features)
            return _score_from_decision(decisions, getattr(model, "classes_", None))
        except Exception:
            pass

    return [None] * _row_count(features)


def _normalize_label(predicted: Any, label_encoder: Any | None) -> str:
    decoded = predicted
    if label_encoder is not None:
        try:
            decoded = label_encoder.inverse_transform([predicted])[0]
        except Exception:
            decoded = predicted

    if isinstance(decoded, (np.integer, int, bool)):
        return "Positive" if int(decoded) == 1 else "Negative"
    if isinstance(decoded, (np.floating, float)):
        if float(decoded) in (0.0, 1.0):
            return "Positive" if float(decoded) == 1.0 else "Negative"

    value = str(decoded).strip().lower()
    if value in {"1", "positive", "pos", "true"} or "posit" in value:
        return "Positive"
    if value in {"0", "negative", "neg", "false"} or "negat" in value:
        return "Negative"
    if not value:
        return "Unknown"
    return str(decoded).strip()


def _apply_neutral_threshold(label: str, score: float | None, model_name: str) -> str:
    if score is None:
        return label

    if model_name == "knn" and KNN_NEUTRAL_MIN <= score <= KNN_NEUTRAL_MAX:
        return "Neutral"

    if model_name == "svm" and SVM_NEUTRAL_MIN <= score <= SVM_NEUTRAL_MAX:
        return "Neutral"

    return label


def _predict_with_optional_vectorizer(
    model: Any,
    texts: list[str],
    artifacts: ModelArtifacts,
    model_name: str,
) -> tuple[list[Any], list[float | None]]:
    try:
        predictions = model.predict(texts)
        scores = _extract_scores(model, texts)
        return list(predictions), scores
    except Exception as direct_error:
        if artifacts.vectorizer is None:
            if isinstance(direct_error, NotFittedError):
                raise ModelServiceError(
                    f"{model_name} gagal dipakai walau tampak pipeline internal ({direct_error}). "
                    "Kemungkinan besar versi scikit-learn/imbalanced-learn di environment tidak cocok "
                    "dengan versi saat model disimpan. Samakan versi dependensi model terlebih dahulu."
                ) from direct_error

            pipeline_hint = ""
            if hasattr(model, "steps") or hasattr(model, "named_steps"):
                pipeline_hint = (
                    " Model ini tampak berupa pipeline internal, jadi cek juga kecocokan versi "
                    "scikit-learn/imbalanced-learn."
                )
            raise ModelServiceError(
                f"{model_name} tidak bisa memprediksi teks mentah secara langsung ({direct_error})."
                f"{pipeline_hint} Jika model Anda classifier-only, tambahkan vectorizer.joblib "
                "atau tfidf_vectorizer.joblib di sentiment_site/models/."
            ) from direct_error

        try:
            vectorized = artifacts.vectorizer.transform(texts)
        except Exception as vectorizer_error:
            raise ModelServiceError(f"Vectorizer gagal mentransformasi teks input: {vectorizer_error}") from vectorizer_error

        try:
            predictions = model.predict(vectorized)
            scores = _extract_scores(model, vectorized)
        except Exception as model_error:
            raise ModelServiceError(
                f"Prediksi {model_name} gagal meskipun sudah melalui vectorization: {model_error}"
            ) from model_error
        return list(predictions), scores


def predict_batch(texts: Iterable[str]) -> list[dict[str, Any]]:
    texts_list = [str(text or "").strip() for text in texts]
    if not texts_list:
        return []

    processed_texts = [preprocess_text(text) for text in texts_list]
    artifacts = _load_artifacts()

    knn_predictions, knn_scores = _predict_with_optional_vectorizer(
        artifacts.knn_model, processed_texts, artifacts, "KNN model"
    )
    svm_predictions, svm_scores = _predict_with_optional_vectorizer(
        artifacts.svm_model, processed_texts, artifacts, "SVM model"
    )

    rows: list[dict[str, Any]] = []
    for index, text in enumerate(texts_list):
        knn_score = knn_scores[index] if index < len(knn_scores) else None
        svm_score = svm_scores[index] if index < len(svm_scores) else None
        knn_label = _apply_neutral_threshold(
            _normalize_label(knn_predictions[index], artifacts.label_encoder),
            knn_score,
            "knn",
        )
        svm_label = _apply_neutral_threshold(
            _normalize_label(svm_predictions[index], artifacts.label_encoder),
            svm_score,
            "svm",
        )
        rows.append(
            {
                "text": text,
                "knn_label": knn_label,
                "knn_score": knn_score,
                "svm_label": svm_label,
                "svm_score": svm_score,
            }
        )
    return rows


def predict_batch_in_chunks(texts: Iterable[str], chunk_size: int = 300) -> list[dict[str, Any]]:
    texts_list = [str(text or "").strip() for text in texts]
    if not texts_list:
        return []

    normalized_chunk_size = max(1, int(chunk_size))
    rows: list[dict[str, Any]] = []
    for start_index in range(0, len(texts_list), normalized_chunk_size):
        chunk = texts_list[start_index : start_index + normalized_chunk_size]
        rows.extend(predict_batch(chunk))
    return rows


def predict_single(text: str) -> dict[str, Any]:
    predictions = predict_batch([text])
    if not predictions:
        raise ModelServiceError("Teks input kosong.")
    return predictions[0]
