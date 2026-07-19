import tempfile
from pathlib import Path
import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase, override_settings
from unittest.mock import patch

from sentiment_app.models import SentimentModelVersion
from sentiment_app.services.model_service import (
    ModelArtifacts,
    ModelServiceError,
    _load_artifacts,
    clear_cache,
    predict_batch,
    predict_batch_in_chunks,
    resolve_model_version_name,
)


class PipelineLikeModel:
    classes_ = np.array([0, 1])

    def predict(self, values):
        outputs = []
        for value in values:
            outputs.append(1 if "good" in str(value).lower() else 0)
        return np.array(outputs)

    def predict_proba(self, values):
        probabilities = []
        for value in values:
            if "good" in str(value).lower():
                probabilities.append([0.1, 0.9])
            else:
                probabilities.append([0.8, 0.2])
        return np.array(probabilities)


class NeedsVectorizerModel:
    classes_ = np.array([0, 1])

    def predict(self, values):
        sample = values[0]
        if isinstance(sample, str) and not sample.startswith("vec::"):
            raise ValueError("Expected vectorized features")
        return np.array([1 for _ in values])

    def predict_proba(self, values):
        return np.array([[0.2, 0.8] for _ in values])


class PrefixVectorizer:
    def transform(self, values):
        return [f"vec::{value}" for value in values]


class NeutralThresholdKNNModel:
    classes_ = np.array([0, 1])

    def predict(self, values):
        return np.array([1 for _ in values])

    def predict_proba(self, values):
        probabilities = []
        for value in values:
            if "netral" in str(value).lower():
                probabilities.append([0.5, 0.5])
            else:
                probabilities.append([0.1, 0.9])
        return np.array(probabilities)


class NeutralThresholdSVMModel:
    classes_ = np.array([0, 1])

    def predict(self, values):
        return np.array([1 for _ in values])

    def decision_function(self, values):
        decisions = []
        for value in values:
            if "netral" in str(value).lower():
                decisions.append(0.05)
            else:
                decisions.append(0.7)
        return np.array(decisions)


class NeutralBoundaryKNNModel:
    classes_ = np.array([0, 1])

    def predict(self, values):
        return np.array([1 for _ in values])

    def predict_proba(self, values):
        probabilities = []
        for value in values:
            text = str(value).lower()
            if "batas atas" in text or "batas-atas" in text:
                probabilities.append([0.4, 0.6])
            elif "lewat atas" in text or "lewat-atas" in text:
                probabilities.append([0.39, 0.61])
            else:
                probabilities.append([0.6, 0.4])
        return np.array(probabilities)


class NeutralBoundarySVMModel:
    classes_ = np.array([0, 1])

    def predict(self, values):
        return np.array([1 for _ in values])

    def decision_function(self, values):
        decisions = []
        for value in values:
            text = str(value).lower()
            if "batas atas" in text or "batas-atas" in text:
                decisions.append(np.log(0.6 / 0.4))
            elif "lewat atas" in text or "lewat-atas" in text:
                decisions.append(np.log(0.61 / 0.39))
            else:
                decisions.append(np.log(0.4 / 0.6))
        return np.array(decisions)


class ModelServiceTests(SimpleTestCase):
    def test_predict_batch_with_direct_pipeline_path(self):
        artifacts = ModelArtifacts(
            knn_model=PipelineLikeModel(),
            svm_model=PipelineLikeModel(),
            vectorizer=None,
            label_encoder=None,
        )

        with patch("sentiment_app.services.model_service._load_artifacts", return_value=artifacts):
            rows = predict_batch(["good day", "bad day"])

        self.assertEqual(rows[0]["knn_label"], "Positive")
        self.assertEqual(rows[1]["svm_label"], "Negative")
        self.assertAlmostEqual(rows[0]["knn_positive_score"], 0.9, places=3)
        self.assertAlmostEqual(rows[0]["knn_negative_score"], 0.1, places=3)
        self.assertGreater(rows[0]["svm_positive_score"], 0.5)
        self.assertAlmostEqual(rows[0]["svm_negative_score"], 0.1, places=3)
        self.assertEqual(rows[0]["combined_label"], "Positive")
        self.assertAlmostEqual(rows[0]["combined_positive_score"], 0.9, places=3)
        self.assertAlmostEqual(rows[0]["combined_negative_score"], 0.1, places=3)
        self.assertEqual(rows[1]["combined_label"], "Negative")
        self.assertAlmostEqual(rows[1]["combined_positive_score"], 0.2, places=3)
        self.assertAlmostEqual(rows[1]["combined_negative_score"], 0.8, places=3)

    def test_predict_batch_uses_vectorizer_fallback(self):
        artifacts = ModelArtifacts(
            knn_model=NeedsVectorizerModel(),
            svm_model=NeedsVectorizerModel(),
            vectorizer=PrefixVectorizer(),
            label_encoder=None,
        )

        with patch("sentiment_app.services.model_service._load_artifacts", return_value=artifacts):
            rows = predict_batch(["HELLO"])

        self.assertEqual(rows[0]["knn_label"], "Positive")
        self.assertEqual(rows[0]["svm_label"], "Positive")
        self.assertAlmostEqual(rows[0]["knn_positive_score"], 0.8, places=3)
        self.assertAlmostEqual(rows[0]["knn_negative_score"], 0.2, places=3)
        self.assertAlmostEqual(rows[0]["svm_positive_score"], 0.8, places=3)
        self.assertAlmostEqual(rows[0]["svm_negative_score"], 0.2, places=3)
        self.assertEqual(rows[0]["combined_label"], "Positive")
        self.assertAlmostEqual(rows[0]["combined_positive_score"], 0.8, places=3)
        self.assertAlmostEqual(rows[0]["combined_negative_score"], 0.2, places=3)

    def test_predict_batch_raises_when_vectorizer_required_but_missing(self):
        artifacts = ModelArtifacts(
            knn_model=NeedsVectorizerModel(),
            svm_model=NeedsVectorizerModel(),
            vectorizer=None,
            label_encoder=None,
        )

        with patch("sentiment_app.services.model_service._load_artifacts", return_value=artifacts):
            with self.assertRaises(ModelServiceError):
                predict_batch(["text"])

    def test_predict_batch_in_chunks_matches_full_prediction(self):
        artifacts = ModelArtifacts(
            knn_model=PipelineLikeModel(),
            svm_model=PipelineLikeModel(),
            vectorizer=None,
            label_encoder=None,
        )

        values = ["good one", "bad one", "good two", "bad two", "good three"]
        with patch("sentiment_app.services.model_service._load_artifacts", return_value=artifacts):
            expected = predict_batch(values)
            actual = predict_batch_in_chunks(values, chunk_size=2)

        self.assertEqual(actual, expected)

    def test_predict_batch_applies_neutral_thresholds_for_knn_and_svm(self):
        artifacts = ModelArtifacts(
            knn_model=NeutralThresholdKNNModel(),
            svm_model=NeutralThresholdSVMModel(),
            vectorizer=None,
            label_encoder=None,
        )

        with patch("sentiment_app.services.model_service._load_artifacts", return_value=artifacts):
            rows = predict_batch(["teks netral", "teks positif"])

        self.assertEqual(rows[0]["knn_label"], "Neutral")
        self.assertEqual(rows[0]["svm_label"], "Neutral")
        self.assertEqual(rows[0]["combined_label"], "Neutral")
        self.assertEqual(rows[1]["knn_label"], "Positive")
        self.assertEqual(rows[1]["svm_label"], "Positive")
        self.assertEqual(rows[1]["combined_label"], "Positive")
        self.assertAlmostEqual(rows[0]["knn_positive_score"], 0.5, places=3)
        self.assertAlmostEqual(rows[0]["knn_negative_score"], 0.5, places=3)
        self.assertAlmostEqual(rows[0]["svm_positive_score"], 1.0 / (1.0 + np.exp(-0.05)), places=3)
        self.assertAlmostEqual(rows[0]["svm_negative_score"], 1.0 - (1.0 / (1.0 + np.exp(-0.05))), places=3)
        self.assertAlmostEqual(rows[1]["svm_positive_score"], 1.0 / (1.0 + np.exp(-0.7)), places=3)
        self.assertAlmostEqual(rows[1]["svm_negative_score"], 1.0 - (1.0 / (1.0 + np.exp(-0.7))), places=3)
        self.assertAlmostEqual(
            rows[0]["combined_positive_score"],
            (0.5 + (1.0 / (1.0 + np.exp(-0.05)))) / 2.0,
            places=3,
        )
        self.assertAlmostEqual(
            rows[1]["combined_positive_score"],
            (0.9 + (1.0 / (1.0 + np.exp(-0.7)))) / 2.0,
            places=3,
        )
        self.assertAlmostEqual(rows[0]["combined_negative_score"], 1.0 - rows[0]["combined_positive_score"], places=3)
        self.assertAlmostEqual(rows[1]["combined_negative_score"], 1.0 - rows[1]["combined_positive_score"], places=3)

    def test_predict_batch_uses_new_neutral_boundaries_of_zero_point_four_to_zero_point_six(self):
        artifacts = ModelArtifacts(
            knn_model=NeutralBoundaryKNNModel(),
            svm_model=NeutralBoundarySVMModel(),
            vectorizer=None,
            label_encoder=None,
        )

        with patch("sentiment_app.services.model_service._load_artifacts", return_value=artifacts):
            rows = predict_batch(["batas-atas", "lewat-atas"])

        self.assertEqual(rows[0]["knn_label"], "Neutral")
        self.assertEqual(rows[0]["svm_label"], "Neutral")
        self.assertEqual(rows[0]["combined_label"], "Neutral")
        self.assertEqual(rows[1]["knn_label"], "Positive")
        self.assertEqual(rows[1]["svm_label"], "Positive")
        self.assertEqual(rows[1]["combined_label"], "Positive")
        self.assertAlmostEqual(rows[0]["knn_positive_score"], 0.6, places=3)
        self.assertAlmostEqual(rows[0]["svm_positive_score"], 0.6, places=3)
        self.assertAlmostEqual(rows[1]["knn_positive_score"], 0.61, places=3)
        self.assertAlmostEqual(rows[1]["svm_positive_score"], 0.61, places=3)


class ModelArtifactStorageFallbackTests(TestCase):
    def test_resolve_model_version_name_prefers_configured_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            media_dir = temp_path / "media"
            version_one = "Resolver Test V1.0"
            version_two = "Resolver Test V2.0"
            with override_settings(
                MEDIA_ROOT=media_dir,
                SENTIMENT_MODELS_DIR=temp_path / "models",
                SENTIMENT_DEFAULT_MODEL_VERSION=version_two,
            ):
                SentimentModelVersion.objects.create(
                    version_name=version_one,
                    knn_model_file=SimpleUploadedFile("knn_model.joblib", b"knn-v1"),
                    svm_model_file=SimpleUploadedFile("svm_linear_model.joblib", b"svm-v1"),
                )
                SentimentModelVersion.objects.create(
                    version_name=version_two,
                    knn_model_file=SimpleUploadedFile("knn_model.joblib", b"knn-v2"),
                    svm_model_file=SimpleUploadedFile("svm_linear_model.joblib", b"svm-v2"),
                )

                self.assertEqual(resolve_model_version_name(""), version_two)
                self.assertEqual(resolve_model_version_name("default"), version_two)

    def test_load_artifacts_falls_back_to_legacy_files_when_db_storage_file_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            media_dir = temp_path / "media"
            models_dir = temp_path / "models"
            version_name = "Sentimen Fallback Test V9.9"
            version_dir = models_dir / version_name
            version_dir.mkdir(parents=True, exist_ok=True)
            (version_dir / "knn_model.joblib").write_bytes(b"legacy-knn")
            (version_dir / "svm_linear_model.joblib").write_bytes(b"legacy-svm")

            with override_settings(MEDIA_ROOT=media_dir, SENTIMENT_MODELS_DIR=models_dir):
                record = SentimentModelVersion.objects.create(
                    version_name=version_name,
                    knn_model_file=SimpleUploadedFile("knn_model.joblib", b"db-knn"),
                    svm_model_file=SimpleUploadedFile("svm_linear_model.joblib", b"db-svm"),
                )
                record.knn_model_file.delete(save=False)
                record.svm_model_file.delete(save=False)
                clear_cache()

                def fake_joblib_load(source):
                    return f"loaded::{Path(str(source)).name}"

                with patch("sentiment_app.services.model_service.joblib.load", side_effect=fake_joblib_load):
                    artifacts = _load_artifacts(version_name)

            self.assertEqual(artifacts.knn_model, "loaded::knn_model.joblib")
            self.assertEqual(artifacts.svm_model, "loaded::svm_linear_model.joblib")
