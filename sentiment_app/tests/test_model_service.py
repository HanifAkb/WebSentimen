import numpy as np
from django.test import SimpleTestCase
from unittest.mock import patch

from sentiment_app.services.model_service import (
    ModelArtifacts,
    ModelServiceError,
    predict_batch,
    predict_batch_in_chunks,
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
        self.assertGreater(rows[0]["svm_score"], 0.5)

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
        self.assertAlmostEqual(rows[0]["knn_score"], 0.8, places=3)

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
