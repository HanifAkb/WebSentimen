from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from django.conf import settings
from sklearn.exceptions import NotFittedError

from .preprocess import preprocess_text


class ModelServiceError(RuntimeError):
    pass


@dataclass
class ModelArtifacts:
    knn_model: Any
    svm_model: Any
    vectorizer: Any | None = None
    label_encoder: Any | None = None


_ARTIFACTS_CACHE: ModelArtifacts | None = None


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
    svm_path = _find_artifact_path("svm_rbf_model.joblib")

    if knn_path is None or svm_path is None:
        missing = []
        if knn_path is None:
            missing.append("knn_model.joblib")
        if svm_path is None:
            missing.append("svm_rbf_model.joblib")
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


def _score_from_decision(decision_values: Any, classes: Any) -> list[float]:
    values = np.asarray(decision_values)
    if values.ndim == 1:
        selected = values
    else:
        positive_index = _positive_class_index(classes)
        if positive_index is None:
            positive_index = 1 if values.shape[1] > 1 else 0
        selected = values[:, positive_index]

    # Keep raw decision score in [-1, 1] range for UI consistency.
    clipped = np.clip(selected, -1.0, 1.0)
    return [float(value) for value in clipped]


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
        rows.append(
            {
                "text": text,
                "knn_label": _normalize_label(knn_predictions[index], artifacts.label_encoder),
                "knn_score": knn_scores[index] if index < len(knn_scores) else None,
                "svm_label": _normalize_label(svm_predictions[index], artifacts.label_encoder),
                "svm_score": svm_scores[index] if index < len(svm_scores) else None,
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
