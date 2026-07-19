import tempfile
from pathlib import Path
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase, override_settings

from sentiment_app.forms import AdminModelEditForm, AdminModelUploadForm, PredictForm, TwitterFetchForm
from sentiment_app.models import SentimentModelVersion


class TwitterFetchFormTests(SimpleTestCase):
    @patch("sentiment_app.forms.available_model_versions", return_value=[("Sentimen V1.0", "Sentimen V1.0")])
    def test_language_choices_hide_english_option(self, _mocked_versions):
        form = TwitterFetchForm()

        self.assertEqual(
            list(form.fields["language"].choices),
            [("in", "Bahasa Indonesia")],
        )
        self.assertEqual(
            list(form.fields["model_version"].choices),
            [("Sentimen V1.0", "Sentimen V1.0")],
        )

    @patch("sentiment_app.forms.available_model_versions", return_value=[("Sentimen V1.0", "Sentimen V1.0")])
    def test_accepts_range_within_limit(self, _mocked_versions):
        form = TwitterFetchForm(
            data={
                "api_key": "dummy",
                "model_version": "Sentimen V1.0",
                "query": "mobil listrik",
                "language": "in",
                "start_date": "01/01/2026",
                "end_date": "07/01/2026",
            }
        )
        self.assertTrue(form.is_valid(), form.errors)

    @patch("sentiment_app.forms.available_model_versions", return_value=[("Sentimen V1.0", "Sentimen V1.0")])
    def test_rejects_when_start_date_after_end_date(self, _mocked_versions):
        form = TwitterFetchForm(
            data={
                "api_key": "dummy",
                "model_version": "Sentimen V1.0",
                "query": "mobil listrik",
                "language": "in",
                "start_date": "10/01/2026",
                "end_date": "09/01/2026",
            }
        )
        self.assertFalse(form.is_valid())
        self.assertIn("Tanggal mulai tidak boleh lebih besar", str(form.non_field_errors()))

    @patch("sentiment_app.forms.available_model_versions", return_value=[("Sentimen V1.0", "Sentimen V1.0")])
    def test_accepts_iso_date_format_from_native_date_input(self, _mocked_versions):
        form = TwitterFetchForm(
            data={
                "api_key": "dummy",
                "model_version": "Sentimen V1.0",
                "query": "mobil listrik",
                "language": "in",
                "start_date": "2026-01-01",
                "end_date": "2026-01-07",
            }
        )
        self.assertTrue(form.is_valid(), form.errors)

    @patch("sentiment_app.forms.available_model_versions", return_value=[("Sentimen V1.0", "Sentimen V1.0")])
    def test_rejects_empty_language_when_only_indonesian_is_allowed(self, _mocked_versions):
        form = TwitterFetchForm(
            data={
                "api_key": "dummy",
                "model_version": "Sentimen V1.0",
                "query": "mobil listrik",
                "language": "",
                "start_date": "01/01/2026",
                "end_date": "07/01/2026",
            }
        )

        self.assertFalse(form.is_valid())
        self.assertIn("language", form.errors)

    @patch("sentiment_app.forms.available_model_versions", return_value=[("Sentimen V1.0", "Sentimen V1.0")])
    def test_rejects_language_outside_dropdown_choices(self, _mocked_versions):
        form = TwitterFetchForm(
            data={
                "api_key": "dummy",
                "model_version": "Sentimen V1.0",
                "query": "mobil listrik",
                "language": "jp",
                "start_date": "01/01/2026",
                "end_date": "07/01/2026",
            }
        )

        self.assertFalse(form.is_valid())
        self.assertIn("language", form.errors)


class PredictFormTests(SimpleTestCase):
    @patch("sentiment_app.forms.available_model_versions", return_value=[("Sentimen V1.0", "Sentimen V1.0")])
    def test_model_version_choices_match_available_versions(self, _mocked_versions):
        form = PredictForm()

        self.assertEqual(
            list(form.fields["model_version"].choices),
            [("Sentimen V1.0", "Sentimen V1.0")],
        )


class AdminModelUploadFormTests(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def _settings(self):
        temp_path = Path(self.temp_dir.name)
        return override_settings(SENTIMENT_MODELS_DIR=temp_path, MEDIA_ROOT=temp_path / "media")

    def test_accepts_new_model_version_with_two_joblib_files(self):
        with self._settings():
            form = AdminModelUploadForm(
                data={"version_name": "Sentimen V2.0"},
                files={
                    "knn_model_file": SimpleUploadedFile("knn_model.joblib", b"knn-bytes"),
                    "svm_model_file": SimpleUploadedFile("svm_linear_model.joblib", b"svm-bytes"),
                },
            )

            self.assertTrue(form.is_valid(), form.errors)

    def test_rejects_existing_model_version_name(self):
        with self._settings():
            SentimentModelVersion.objects.create(
                version_name="Sentimen V2.0",
                knn_model_file=SimpleUploadedFile("knn_model.joblib", b"knn-bytes"),
                svm_model_file=SimpleUploadedFile("svm_linear_model.joblib", b"svm-bytes"),
            )
            form = AdminModelUploadForm(
                data={"version_name": "Sentimen V2.0"},
                files={
                    "knn_model_file": SimpleUploadedFile("knn_model.joblib", b"knn-bytes"),
                    "svm_model_file": SimpleUploadedFile("svm_linear_model.joblib", b"svm-bytes"),
                },
            )

            self.assertFalse(form.is_valid())
            self.assertIn("version_name", form.errors)

    def test_rejects_non_joblib_upload(self):
        with self._settings():
            form = AdminModelUploadForm(
                data={"version_name": "Sentimen V2.0"},
                files={
                    "knn_model_file": SimpleUploadedFile("knn_model.txt", b"knn-bytes"),
                    "svm_model_file": SimpleUploadedFile("svm_linear_model.joblib", b"svm-bytes"),
                },
            )

            self.assertFalse(form.is_valid())
            self.assertIn("knn_model_file", form.errors)

    def test_edit_form_allows_same_existing_version_name(self):
        with self._settings():
            SentimentModelVersion.objects.create(
                version_name="Sentimen V2.0",
                knn_model_file=SimpleUploadedFile("knn_model.joblib", b"knn-bytes"),
                svm_model_file=SimpleUploadedFile("svm_linear_model.joblib", b"svm-bytes"),
            )
            form = AdminModelEditForm(
                data={"version_name": "Sentimen V2.0"},
                files={},
                existing_version_name="Sentimen V2.0",
            )

            self.assertTrue(form.is_valid(), form.errors)

    def test_edit_form_rejects_rename_to_existing_other_version(self):
        with self._settings():
            SentimentModelVersion.objects.create(
                version_name="Sentimen V2.0",
                knn_model_file=SimpleUploadedFile("knn_model.joblib", b"knn-bytes"),
                svm_model_file=SimpleUploadedFile("svm_linear_model.joblib", b"svm-bytes"),
            )
            SentimentModelVersion.objects.create(
                version_name="Sentimen V3.0",
                knn_model_file=SimpleUploadedFile("knn_model.joblib", b"knn-bytes"),
                svm_model_file=SimpleUploadedFile("svm_linear_model.joblib", b"svm-bytes"),
            )
            form = AdminModelEditForm(
                data={"version_name": "Sentimen V3.0"},
                files={},
                existing_version_name="Sentimen V2.0",
            )

            self.assertFalse(form.is_valid())
            self.assertIn("version_name", form.errors)
