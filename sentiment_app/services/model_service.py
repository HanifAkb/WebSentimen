from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from django.conf import settings
from sklearn.exceptions import NotFittedError

from sentiment_app.models import SentimentModelVersion

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


DEFAULT_MODEL_VERSION_KEY = "default"
_ARTIFACTS_CACHE: dict[str, ModelArtifacts] = {}
KNN_NEUTRAL_MIN = 0.4
KNN_NEUTRAL_MAX = 0.6
SVM_NEUTRAL_MIN = 0.4
SVM_NEUTRAL_MAX = 0.6
SOFT_VOTING_KNN_WEIGHT = 0.5
SOFT_VOTING_SVM_WEIGHT = 0.5
SOFT_VOTING_NEUTRAL_MIN = 0.4
SOFT_VOTING_NEUTRAL_MAX = 0.6


def _load_stopwords() -> set[str]:
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is not None:
        return _STOPWORDS_CACHE

    models_dir = _models_dir()
    candidates = [
        models_dir / "stopwords-id.txt",
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

    _STOPWORDS_CACHE = set()
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

    _SLANG_MAP_CACHE = {}
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
    _ARTIFACTS_CACHE = {}


def models_dir_path() -> Path:
    return Path(getattr(settings, "SENTIMENT_MODELS_DIR", settings.BASE_DIR / "sentiment_site" / "models"))


def _models_dir() -> Path:
    return models_dir_path()


def _has_required_model_files(directory: Path) -> bool:
    if not directory.exists() or not directory.is_dir():
        return False
    if not (directory / "knn_model.joblib").exists():
        return False
    return any((directory / candidate).exists() for candidate in ("svm_linear_model.joblib", "svm_rbf_model.joblib"))


def _legacy_available_model_versions() -> list[tuple[str, str]]:
    models_dir = _models_dir()
    versions: list[tuple[str, str]] = []
    if models_dir.exists():
        for child in sorted(models_dir.iterdir(), key=lambda path: path.name.lower()):
            if child.is_dir() and _has_required_model_files(child):
                versions.append((child.name, child.name))
    if versions:
        return versions
    if _has_required_model_files(models_dir):
        return [(DEFAULT_MODEL_VERSION_KEY, "Default")]
    return []


def _db_model_versions() -> list[SentimentModelVersion]:
    return list(SentimentModelVersion.objects.order_by("version_name", "id"))


def available_model_versions() -> list[tuple[str, str]]:
    versions: list[tuple[str, str]] = []
    seen_names: set[str] = set()
    for version_name, version_label in _legacy_available_model_versions():
        if version_name in seen_names:
            continue
        versions.append((version_name, version_label))
        seen_names.add(version_name)
    for record in _db_model_versions():
        if record.version_name in seen_names:
            continue
        versions.append((record.version_name, record.version_name))
        seen_names.add(record.version_name)
    return versions


def resolve_model_version_name(model_version: str | None = None) -> str:
    selected_version = str(model_version or "").strip()
    if selected_version and selected_version != DEFAULT_MODEL_VERSION_KEY:
        return selected_version

    versions = available_model_versions()
    if not versions:
        return selected_version

    configured_default = str(getattr(settings, "SENTIMENT_DEFAULT_MODEL_VERSION", "") or "").strip()
    if configured_default:
        for version_name, _version_label in versions:
            if version_name == configured_default:
                return version_name

    return str(versions[0][0] or "").strip()


def _resolve_model_directory(model_version: str | None = None) -> Path:
    models_dir = _models_dir()
    selected_version = str(model_version or "").strip()
    if selected_version and selected_version != DEFAULT_MODEL_VERSION_KEY:
        candidate = models_dir / selected_version
        if candidate.exists() and candidate.is_dir():
            return candidate
        raise ModelServiceError(f"Versi model tidak ditemukan: {selected_version}")

    if _has_required_model_files(models_dir):
        return models_dir

    versions = available_model_versions()
    if not versions:
        return models_dir

    configured_default = str(getattr(settings, "SENTIMENT_DEFAULT_MODEL_VERSION", "") or "").strip()
    if configured_default:
        for key, _label in versions:
            if key == configured_default:
                return models_dir / key

    return models_dir / versions[0][0]


def _find_artifact_path(filename: str, model_version: str | None = None) -> Path | None:
    try:
        primary_dir = _resolve_model_directory(model_version)
    except ModelServiceError:
        primary_dir = None
    primary = primary_dir / filename if primary_dir is not None else None
    fallback = Path(settings.BASE_DIR) / filename
    if primary is not None and primary.exists():
        return primary
    root_path = _models_dir() / filename
    if root_path.exists():
        return root_path
    if fallback.exists():
        return fallback
    return None


def _resolve_db_model_record(model_version: str | None = None) -> SentimentModelVersion | None:
    selected_version = str(model_version or "").strip()
    if selected_version and selected_version != DEFAULT_MODEL_VERSION_KEY:
        return SentimentModelVersion.objects.filter(version_name=selected_version).first()

    configured_default = str(getattr(settings, "SENTIMENT_DEFAULT_MODEL_VERSION", "") or "").strip()
    if configured_default:
        configured_record = SentimentModelVersion.objects.filter(version_name=configured_default).first()
        if configured_record is not None:
            return configured_record

    return SentimentModelVersion.objects.order_by("version_name", "id").first()


def _stored_file_exists(file_field) -> bool:
    if not file_field:
        return False
    file_name = str(getattr(file_field, "name", "") or "").strip()
    if not file_name:
        return False
    try:
        return bool(file_field.storage.exists(file_name))
    except Exception:
        return False


def _load_joblib_file(file_field, label: str):
    if not file_field:
        return None
    try:
        with file_field.open("rb") as uploaded_file:
            return joblib.load(uploaded_file)
    except Exception as exc:
        raise ModelServiceError(f"Gagal memuat {label}: {exc}") from exc


def _load_artifacts(model_version: str | None = None) -> ModelArtifacts:
    global _ARTIFACTS_CACHE
    selected_version = str(model_version or "").strip()

    try:
        model_directory = _resolve_model_directory(selected_version or None)
    except ModelServiceError:
        model_directory = None

    if model_directory is not None and _has_required_model_files(model_directory):
        cache_key = str(model_directory.resolve())
        if cache_key in _ARTIFACTS_CACHE:
            return _ARTIFACTS_CACHE[cache_key]

        knn_path = _find_artifact_path("knn_model.joblib", selected_version or None)
        svm_candidates = ["svm_linear_model.joblib", "svm_rbf_model.joblib"]
        svm_path = next((path for path in (_find_artifact_path(name, selected_version or None) for name in svm_candidates) if path is not None), None)

        if knn_path is None or svm_path is None:
            missing = []
            if knn_path is None:
                missing.append("knn_model.joblib")
            if svm_path is None:
                missing.append(" / ".join(svm_candidates))
            raise ModelServiceError(
                "File model wajib tidak ditemukan: "
                f"{', '.join(missing)}. Letakkan file tersebut di {model_directory}."
            )

        try:
            knn_model = joblib.load(knn_path)
            svm_model = joblib.load(svm_path)
        except Exception as exc:
            raise ModelServiceError(f"Gagal memuat file model: {exc}") from exc

        vectorizer = None
        for vectorizer_name in ("vectorizer.joblib", "tfidf_vectorizer.joblib"):
            vectorizer_path = _find_artifact_path(vectorizer_name, selected_version or None)
            if vectorizer_path is not None:
                try:
                    vectorizer = joblib.load(vectorizer_path)
                except Exception as exc:
                    raise ModelServiceError(f"Gagal memuat {vectorizer_name}: {exc}") from exc
                break

        label_encoder = None
        label_encoder_path = _find_artifact_path("label_encoder.joblib", selected_version or None)
        if label_encoder_path is not None:
            try:
                label_encoder = joblib.load(label_encoder_path)
            except Exception as exc:
                raise ModelServiceError(f"Gagal memuat label_encoder.joblib: {exc}") from exc

        artifacts = ModelArtifacts(
            knn_model=knn_model,
            svm_model=svm_model,
            vectorizer=vectorizer,
            label_encoder=label_encoder,
        )
        _ARTIFACTS_CACHE[cache_key] = artifacts
        return artifacts

    db_record = _resolve_db_model_record(model_version)
    if db_record is not None:
        db_version_name = str(db_record.version_name or "").strip()
        if _stored_file_exists(db_record.knn_model_file) and _stored_file_exists(db_record.svm_model_file):
            cache_key = f"db:{db_record.pk}:{db_record.updated_at.isoformat()}"
            if cache_key in _ARTIFACTS_CACHE:
                return _ARTIFACTS_CACHE[cache_key]

            knn_model = _load_joblib_file(db_record.knn_model_file, "file model KNN")
            svm_model = _load_joblib_file(db_record.svm_model_file, "file model SVM")

            vectorizer = None
            for vectorizer_name in ("vectorizer.joblib", "tfidf_vectorizer.joblib"):
                vectorizer_path = _find_artifact_path(vectorizer_name, model_version)
                if vectorizer_path is not None:
                    try:
                        vectorizer = joblib.load(vectorizer_path)
                    except Exception as exc:
                        raise ModelServiceError(f"Gagal memuat {vectorizer_name}: {exc}") from exc
                    break

            label_encoder = None
            label_encoder_path = _find_artifact_path("label_encoder.joblib", model_version)
            if label_encoder_path is not None:
                try:
                    label_encoder = joblib.load(label_encoder_path)
                except Exception as exc:
                    raise ModelServiceError(f"Gagal memuat label_encoder.joblib: {exc}") from exc

            artifacts = ModelArtifacts(
                knn_model=knn_model,
                svm_model=svm_model,
                vectorizer=vectorizer,
                label_encoder=label_encoder,
            )
            _ARTIFACTS_CACHE[cache_key] = artifacts
            return artifacts
        if not selected_version:
            selected_version = db_version_name

    try:
        model_directory = _resolve_model_directory(selected_version or None)
    except ModelServiceError as exc:
        if db_record is not None:
            version_label = str(db_record.version_name or selected_version or "").strip() or "default"
            raise ModelServiceError(
                f"File model untuk versi {version_label} tidak ditemukan di storage aktif, dan fallback legacy juga tidak tersedia."
            ) from exc
        raise
    cache_key = str(model_directory.resolve())
    if cache_key in _ARTIFACTS_CACHE:
        return _ARTIFACTS_CACHE[cache_key]

    knn_path = _find_artifact_path("knn_model.joblib", selected_version or None)
    svm_candidates = ["svm_linear_model.joblib", "svm_rbf_model.joblib"]
    svm_path = next((path for path in (_find_artifact_path(name, selected_version or None) for name in svm_candidates) if path is not None), None)

    if knn_path is None or svm_path is None:
        missing = []
        if knn_path is None:
            missing.append("knn_model.joblib")
        if svm_path is None:
            missing.append(" / ".join(svm_candidates))
        raise ModelServiceError(
            "File model wajib tidak ditemukan: "
            f"{', '.join(missing)}. Letakkan file tersebut di {model_directory}."
        )

    try:
        knn_model = joblib.load(knn_path)
        svm_model = joblib.load(svm_path)
    except Exception as exc:
        raise ModelServiceError(f"Gagal memuat file model: {exc}") from exc

    vectorizer = None
    for vectorizer_name in ("vectorizer.joblib", "tfidf_vectorizer.joblib"):
        vectorizer_path = _find_artifact_path(vectorizer_name, selected_version or None)
        if vectorizer_path is not None:
            try:
                vectorizer = joblib.load(vectorizer_path)
            except Exception as exc:
                raise ModelServiceError(f"Gagal memuat {vectorizer_name}: {exc}") from exc
            break

    label_encoder = None
    label_encoder_path = _find_artifact_path("label_encoder.joblib", selected_version or None)
    if label_encoder_path is not None:
        try:
            label_encoder = joblib.load(label_encoder_path)
        except Exception as exc:
            raise ModelServiceError(f"Gagal memuat label_encoder.joblib: {exc}") from exc

    artifacts = ModelArtifacts(
        knn_model=knn_model,
        svm_model=svm_model,
        vectorizer=vectorizer,
        label_encoder=label_encoder,
    )
    _ARTIFACTS_CACHE[cache_key] = artifacts
    return artifacts


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


def _probability_pairs_from_positive_scores(positive_scores: Any) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for raw_score in np.asarray(positive_scores, dtype=float):
        positive_score = float(np.clip(raw_score, 0.0, 1.0))
        negative_score = float(np.clip(1.0 - positive_score, 0.0, 1.0))
        pairs.append((negative_score, positive_score))
    return pairs


def _probability_pairs_from_proba(probabilities: Any, classes: Any) -> list[tuple[float, float]]:
    values = np.asarray(probabilities)
    if values.ndim == 1:
        return _probability_pairs_from_positive_scores(values)

    positive_index = _positive_class_index(classes)
    if positive_index is None:
        positive_index = 1 if values.shape[1] > 1 else 0

    negative_index = 0 if positive_index != 0 else (1 if values.shape[1] > 1 else None)
    pairs: list[tuple[float, float]] = []
    for row in values:
        positive_score = float(row[positive_index])
        if negative_index is None:
            negative_score = float(np.clip(1.0 - positive_score, 0.0, 1.0))
        else:
            negative_score = float(row[negative_index])
        pairs.append((negative_score, positive_score))
    return pairs


def _sigmoid(values: Any) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _probability_pairs_from_decision(decision_values: Any, classes: Any) -> list[tuple[float, float]]:
    values = np.asarray(decision_values)
    if values.ndim == 1:
        selected = values
    else:
        positive_index = _positive_class_index(classes)
        if positive_index is None:
            positive_index = 1 if values.shape[1] > 1 else 0
        selected = values[:, positive_index]

    probabilities = _sigmoid(selected)
    return _probability_pairs_from_positive_scores(probabilities)


def _extract_probability_pairs(model: Any, features: Any) -> list[tuple[float | None, float | None]]:
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(features)
            return _probability_pairs_from_proba(probabilities, getattr(model, "classes_", None))
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        try:
            decisions = model.decision_function(features)
            return _probability_pairs_from_decision(decisions, getattr(model, "classes_", None))
        except Exception:
            pass

    return [(None, None)] * _row_count(features)


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


def _combine_soft_weighted_vote(
    knn_negative_score: float | None,
    knn_positive_score: float | None,
    svm_negative_score: float | None,
    svm_positive_score: float | None,
) -> tuple[str, float | None, float | None]:
    weighted_positive_scores: list[tuple[float, float]] = []
    weighted_negative_scores: list[tuple[float, float]] = []
    if knn_positive_score is not None:
        weighted_positive_scores.append((float(knn_positive_score), SOFT_VOTING_KNN_WEIGHT))
    if knn_negative_score is not None:
        weighted_negative_scores.append((float(knn_negative_score), SOFT_VOTING_KNN_WEIGHT))
    if svm_positive_score is not None:
        weighted_positive_scores.append((float(svm_positive_score), SOFT_VOTING_SVM_WEIGHT))
    if svm_negative_score is not None:
        weighted_negative_scores.append((float(svm_negative_score), SOFT_VOTING_SVM_WEIGHT))

    if not weighted_positive_scores and not weighted_negative_scores:
        return "Unknown", None, None

    positive_total_weight = sum(weight for _, weight in weighted_positive_scores)
    negative_total_weight = sum(weight for _, weight in weighted_negative_scores)

    combined_positive_score = (
        sum(score * weight for score, weight in weighted_positive_scores) / positive_total_weight
        if positive_total_weight > 0
        else None
    )
    combined_negative_score = (
        sum(score * weight for score, weight in weighted_negative_scores) / negative_total_weight
        if negative_total_weight > 0
        else None
    )

    if combined_positive_score is None and combined_negative_score is None:
        return "Unknown", None, None
    if combined_positive_score is None:
        if combined_negative_score is not None and combined_negative_score > SOFT_VOTING_NEUTRAL_MAX:
            return "Negative", None, float(combined_negative_score)
        return "Neutral", None, float(combined_negative_score) if combined_negative_score is not None else None
    if combined_negative_score is None:
        if combined_positive_score > SOFT_VOTING_NEUTRAL_MAX:
            return "Positive", float(combined_positive_score), None
        return "Neutral", float(combined_positive_score), None

    if combined_positive_score >= combined_negative_score:
        if combined_positive_score > SOFT_VOTING_NEUTRAL_MAX:
            return "Positive", float(combined_positive_score), float(combined_negative_score)
        return "Neutral", float(combined_positive_score), float(combined_negative_score)
    if combined_negative_score > SOFT_VOTING_NEUTRAL_MAX:
        return "Negative", float(combined_positive_score), float(combined_negative_score)
    return "Neutral", float(combined_positive_score), float(combined_negative_score)


def _predict_with_optional_vectorizer(
    model: Any,
    texts: list[str],
    artifacts: ModelArtifacts,
    model_name: str,
) -> tuple[list[Any], list[tuple[float | None, float | None]]]:
    try:
        predictions = model.predict(texts)
        probability_pairs = _extract_probability_pairs(model, texts)
        return list(predictions), probability_pairs
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
            probability_pairs = _extract_probability_pairs(model, vectorized)
        except Exception as model_error:
            raise ModelServiceError(
                f"Prediksi {model_name} gagal meskipun sudah melalui vectorization: {model_error}"
            ) from model_error
        return list(predictions), probability_pairs


def predict_batch(texts: Iterable[str], model_version: str | None = None) -> list[dict[str, Any]]:
    texts_list = [str(text or "").strip() for text in texts]
    if not texts_list:
        return []

    processed_texts = [preprocess_text(text) for text in texts_list]
    artifacts = _load_artifacts(model_version)

    knn_predictions, knn_probability_pairs = _predict_with_optional_vectorizer(
        artifacts.knn_model, processed_texts, artifacts, "KNN model"
    )
    svm_predictions, svm_probability_pairs = _predict_with_optional_vectorizer(
        artifacts.svm_model, processed_texts, artifacts, "SVM model"
    )

    rows: list[dict[str, Any]] = []
    for index, text in enumerate(texts_list):
        knn_negative_score, knn_positive_score = (
            knn_probability_pairs[index] if index < len(knn_probability_pairs) else (None, None)
        )
        svm_negative_score, svm_positive_score = (
            svm_probability_pairs[index] if index < len(svm_probability_pairs) else (None, None)
        )
        knn_label = _apply_neutral_threshold(
            _normalize_label(knn_predictions[index], artifacts.label_encoder),
            knn_positive_score,
            "knn",
        )
        svm_label = _apply_neutral_threshold(
            _normalize_label(svm_predictions[index], artifacts.label_encoder),
            svm_positive_score,
            "svm",
        )
        combined_label, combined_positive_score, combined_negative_score = _combine_soft_weighted_vote(
            knn_negative_score,
            knn_positive_score,
            svm_negative_score,
            svm_positive_score,
        )
        rows.append(
            {
                "text": text,
                "knn_label": knn_label,
                "knn_positive_score": knn_positive_score,
                "knn_negative_score": knn_negative_score,
                "svm_label": svm_label,
                "svm_positive_score": svm_positive_score,
                "svm_negative_score": svm_negative_score,
                "combined_label": combined_label,
                "combined_positive_score": combined_positive_score,
                "combined_negative_score": combined_negative_score,
            }
        )
    return rows


def predict_batch_in_chunks(
    texts: Iterable[str],
    chunk_size: int = 300,
    model_version: str | None = None,
) -> list[dict[str, Any]]:
    texts_list = [str(text or "").strip() for text in texts]
    if not texts_list:
        return []

    normalized_chunk_size = max(1, int(chunk_size))
    rows: list[dict[str, Any]] = []
    for start_index in range(0, len(texts_list), normalized_chunk_size):
        chunk = texts_list[start_index : start_index + normalized_chunk_size]
        rows.extend(predict_batch(chunk, model_version=model_version))
    return rows


def predict_single(text: str, model_version: str | None = None) -> dict[str, Any]:
    predictions = predict_batch([text], model_version=model_version)
    if not predictions:
        raise ModelServiceError("Teks input kosong.")
    return predictions[0]
