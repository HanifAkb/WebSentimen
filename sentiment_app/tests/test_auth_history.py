import io
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from sentiment_app.models import PredictionHistory, ScrapeHistory, ScrapeTempChunk, SentimentModelVersion
from sentiment_app.services.scraping_service import TwitterRateLimitError, TwitterTimeoutError


class AuthAndHistoryTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_superuser(
            username="admin",
            email="admin@example.com",
            password="AdminPass123!",
        )
        self.user = User.objects.create_user(
            username="member",
            email="member@example.com",
            password="MemberPass123!",
        )
        self.other_user = User.objects.create_user(
            username="other",
            email="other@example.com",
            password="OtherPass123!",
        )

    @staticmethod
    def _run_background_job_inline(_job_name, target, *args, **kwargs):
        target(*args, **kwargs)

    def test_anonymous_user_redirected_to_login(self):
        response = self.client.get(reverse("twitter_fetch"))
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("login"), response.url)

    def test_login_sql_injection_payload_fails(self):
        response = self.client.post(
            reverse("login"),
            {
                "username": "admin' OR '1'='1",
                "password": "anything",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_register_route_is_removed(self):
        self.client.force_login(self.user)
        response = self.client.get("/register/")
        self.assertEqual(response.status_code, 404)

    def test_admin_route_is_available(self):
        self.assertEqual(reverse("admin:index"), "/admin/")

    def test_custom_admin_requires_superuser(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("admin:index"))
        self.assertEqual(response.status_code, 403)

    def test_custom_admin_shows_users_and_history_datasets(self):
        scrape_history = ScrapeHistory.objects.create(
            user=self.other_user,
            query="dataset scraping",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-02",
            tweet_count=7,
            rows=[],
        )
        prediction_history = PredictionHistory.objects.create(
            user=self.other_user,
            source_name="dataset.csv",
            text_column="text",
            sample_count=3,
            rows=[],
        )

        self.client.force_login(self.admin)
        response = self.client.get(reverse("admin:index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Admin Panel")
        self.assertContains(response, "Tambah User")
        self.assertContains(response, "Hasil CSV/TXT")
        self.assertContains(response, "Hasil Pengumpulan Data X")
        self.assertNotContains(response, "Dataset PredictionHistory")
        self.assertNotContains(response, "Dataset ScrapeHistory")
        self.assertContains(response, "<th>Username</th>", html=True, count=2)
        self.assertContains(response, "<th>No.</th>", html=True)
        self.assertContains(response, "<th>Dibuat Pada</th>", html=True, count=2)
        self.assertContains(response, "<th>Kueri Pencarian</th>", html=True)
        self.assertNotContains(response, "<th>Kueri</th>", html=True)
        self.assertNotContains(response, "<th>ID</th>", html=True)
        self.assertContains(response, "Administrator")
        self.assertContains(response, "Aktif")
        self.assertNotContains(response, "<th>Peran</th>", html=True)
        self.assertNotContains(response, "<th>Role</th>", html=True)
        self.assertContains(response, "<th>File Type</th>", html=True)
        self.assertNotContains(response, "<th>Bahasa</th>", html=True)
        self.assertNotContains(response, "<th>Tipe</th>", html=True)
        self.assertNotContains(response, "<th>Kolom Teks</th>", html=True)
        self.assertNotContains(response, '<th class="text-center">Kolom Teks</th>', html=True)
        self.assertContains(response, '<th class="text-center">Jml. Data</th>', html=True)
        self.assertContains(response, '<th class="text-center">Jml. Tweet</th>', html=True)
        self.assertNotContains(response, "<th>Aksi</th>", html=True)
        self.assertContains(response, 'class="btn btn-sm admin-user-edit-btn"', count=3)
        self.assertContains(response, 'class="btn btn-sm admin-user-delete-btn"', count=3)
        self.assertContains(response, '<td class="text-center">3</td>', html=True)
        self.assertContains(response, '<td class="text-center">7</td>', html=True)
        self.assertContains(response, '<span class="badge text-bg-primary">CSV</span>', html=True)
        self.assertNotContains(response, "Input manual")
        self.assertContains(response, "Login Terakhir")
        self.assertContains(response, "Username: member")
        self.assertNotContains(response, "No. 1")
        self.assertContains(response, "member")
        self.assertContains(response, "dataset scraping")
        self.assertContains(response, "1 - 2 Januari 2026")
        self.assertContains(response, "dataset.csv")
        self.assertContains(response, 'action="/admin/history/delete-selected/"', html=False, count=2)
        self.assertContains(response, "admin-prediction-delete-selected-button")
        self.assertContains(response, "admin-scrape-delete-selected-button")
        self.assertNotContains(response, reverse("admin:prediction_history_edit", args=[prediction_history.id]))
        self.assertNotContains(response, reverse("admin:scrape_history_edit", args=[scrape_history.id]))
        self.assertNotContains(response, reverse("admin:prediction_history_delete", args=[prediction_history.id]))
        self.assertNotContains(response, reverse("admin:scrape_history_delete", args=[scrape_history.id]))

    def test_prediction_history_file_type_badge_class_depends_on_extension(self):
        csv_history = PredictionHistory(
            user=self.user,
            source_name="dataset.csv",
        )
        txt_history = PredictionHistory(
            user=self.user,
            source_name="dataset.txt",
        )

        self.assertEqual(csv_history.file_type_label, "CSV")
        self.assertEqual(csv_history.file_type_badge_class, "text-bg-primary")
        self.assertEqual(txt_history.file_type_label, "TXT")
        self.assertEqual(txt_history.file_type_badge_class, "text-bg-info")

    def test_custom_admin_can_create_edit_and_delete_user(self):
        self.client.force_login(self.admin)
        create_response = self.client.post(
            reverse("admin:user_add"),
            {
                "username": "created_user",
                "full_name": "Created User",
                "email": "created@example.com",
                "is_active": "on",
                "role": "staff",
                "password1": "CreatedPass123!",
                "password2": "CreatedPass123!",
            },
        )
        self.assertEqual(create_response.status_code, 302)
        self.assertEqual(create_response["Location"], reverse("admin:index"))
        created_user = User.objects.get(username="created_user")
        self.assertEqual(created_user.get_full_name(), "Created User")
        self.assertTrue(created_user.is_active)
        self.assertTrue(created_user.is_staff)
        self.assertFalse(created_user.is_superuser)

        edit_response = self.client.post(
            reverse("admin:user_edit", args=[created_user.id]),
            {
                "username": "edited_user",
                "full_name": "Edited Name",
                "email": "edited@example.com",
                "is_active": "on",
                "role": "administrator",
                "password1": "EditedPass123!",
                "password2": "EditedPass123!",
            },
        )
        self.assertEqual(edit_response.status_code, 302)
        created_user.refresh_from_db()
        self.assertEqual(created_user.username, "edited_user")
        self.assertEqual(created_user.get_full_name(), "Edited Name")
        self.assertTrue(created_user.is_active)
        self.assertTrue(created_user.is_superuser)
        self.assertTrue(created_user.is_staff)
        self.assertTrue(created_user.check_password("EditedPass123!"))

        delete_response = self.client.post(reverse("admin:user_delete", args=[created_user.id]))
        self.assertEqual(delete_response.status_code, 302)
        self.assertFalse(User.objects.filter(id=created_user.id).exists())

    def test_custom_admin_edit_user_page_hides_delete_button(self):
        self.client.force_login(self.admin)
        response = self.client.get(reverse("admin:user_edit", args=[self.user.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Edit User")
        self.assertContains(response, "Edit data akun")
        self.assertContains(response, "Peran")
        self.assertContains(response, 'name="is_active"')
        self.assertNotContains(response, 'name="is_staff"')
        self.assertNotContains(response, 'name="is_superuser"')
        self.assertNotContains(response, "Hapus User")
        self.assertNotContains(response, reverse("admin:user_delete", args=[self.user.id]))

    def test_custom_admin_create_user_page_uses_add_description(self):
        self.client.force_login(self.admin)
        response = self.client.get(reverse("admin:user_add"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Tambah data akun")
        self.assertNotContains(response, "Atur data akun, status aktif, staff, dan Administrator.")

    def test_custom_admin_requires_full_name_when_creating_user(self):
        self.client.force_login(self.admin)
        response = self.client.post(
            reverse("admin:user_add"),
            {
                "username": "missing_name",
                "full_name": "",
                "email": "missing-name@example.com",
                "is_active": "on",
                "role": "staff",
                "password1": "CreatedPass123!",
                "password2": "CreatedPass123!",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertFormError(response.context["form"], "full_name", "Bidang ini tidak boleh kosong.")
        self.assertFalse(User.objects.filter(username="missing_name").exists())

    def test_custom_admin_shows_model_management_option(self):
        self.client.force_login(self.admin)
        version_name = "Sentimen Test V1.0"
        with tempfile.TemporaryDirectory() as temp_dir:
            media_dir = Path(temp_dir) / "media"
            with self.settings(SENTIMENT_MODELS_DIR=Path(temp_dir), MEDIA_ROOT=media_dir):
                SentimentModelVersion.objects.create(
                    version_name=version_name,
                    knn_model_file=SimpleUploadedFile("knn_model.joblib", b"knn"),
                    svm_model_file=SimpleUploadedFile("svm_linear_model.joblib", b"svm"),
                )
                response = self.client.get(reverse("admin:index"))

            self.assertEqual(response.status_code, 200)
            self.assertContains(response, "Tambah Model")
            self.assertContains(response, "Versi Model")
            self.assertContains(response, "File KNN")
            self.assertContains(response, "File SVM")
            self.assertContains(response, reverse("admin:model_edit", args=[version_name]))
            self.assertContains(response, reverse("admin:model_delete", args=[version_name]))

    def test_custom_admin_can_upload_new_model_version(self):
        self.client.force_login(self.admin)
        with tempfile.TemporaryDirectory() as temp_dir:
            media_dir = Path(temp_dir) / "media"
            with self.settings(SENTIMENT_MODELS_DIR=Path(temp_dir), MEDIA_ROOT=media_dir):
                response = self.client.post(
                    reverse("admin:model_add"),
                    {
                        "version_name": "Sentimen V2.0",
                        "knn_model_file": SimpleUploadedFile("knn_model.joblib", b"knn-bytes"),
                        "svm_model_file": SimpleUploadedFile("svm_linear_model.joblib", b"svm-bytes"),
                    },
                )

                self.assertEqual(response.status_code, 302)
                self.assertEqual(response["Location"], reverse("admin:index"))
                record = SentimentModelVersion.objects.get(version_name="Sentimen V2.0")
                self.assertTrue((media_dir / record.knn_model_file.name).exists())
                self.assertTrue((media_dir / record.svm_model_file.name).exists())
                self.assertEqual(record.knn_model_file.name, "sentiment_models/sentimen-v20/knn_model.joblib")
                self.assertEqual(record.svm_model_file.name, "sentiment_models/sentimen-v20/svm_model.joblib")

    def test_custom_admin_rejects_invalid_model_upload_extension(self):
        self.client.force_login(self.admin)
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.settings(SENTIMENT_MODELS_DIR=Path(temp_dir), MEDIA_ROOT=Path(temp_dir) / "media"):
                response = self.client.post(
                    reverse("admin:model_add"),
                    {
                        "version_name": "Sentimen V2.0",
                        "knn_model_file": SimpleUploadedFile("knn_model.txt", b"knn-bytes"),
                        "svm_model_file": SimpleUploadedFile("svm_linear_model.joblib", b"svm-bytes"),
                    },
                )

                self.assertEqual(response.status_code, 200)
                self.assertContains(response, "File model KNN harus berupa file .joblib.")
                self.assertFalse((Path(temp_dir) / "Sentimen V2.0").exists())

    def test_custom_admin_can_edit_model_version(self):
        self.client.force_login(self.admin)
        old_version_name = "Sentimen Test V1.0"
        new_version_name = "Sentimen Test V2.0"
        with tempfile.TemporaryDirectory() as temp_dir:
            media_dir = Path(temp_dir) / "media"
            history = ScrapeHistory.objects.create(
                user=self.user,
                query="mobil listrik",
                model_version=old_version_name,
                language="in",
                start_date="2026-01-01",
                end_date="2026-01-02",
                tweet_count=1,
                rows=[],
            )
            prediction_history = PredictionHistory.objects.create(
                user=self.user,
                source_name="hasil.csv",
                model_version=old_version_name,
                text_column="text",
                sample_count=1,
                rows=[],
            )
            with self.settings(SENTIMENT_MODELS_DIR=Path(temp_dir), MEDIA_ROOT=media_dir):
                record = SentimentModelVersion.objects.create(
                    version_name=old_version_name,
                    knn_model_file=SimpleUploadedFile("knn_model.joblib", b"old-knn"),
                    svm_model_file=SimpleUploadedFile("svm_linear_model.joblib", b"old-svm"),
                )
                response = self.client.post(
                    reverse("admin:model_edit", args=[old_version_name]),
                    {
                        "version_name": new_version_name,
                        "knn_model_file": SimpleUploadedFile("knn_model.joblib", b"new-knn"),
                        "svm_model_file": SimpleUploadedFile("svm_linear_model.joblib", b"new-svm"),
                    },
                )

            self.assertEqual(response.status_code, 302)
            record.refresh_from_db()
            self.assertEqual(record.version_name, new_version_name)
            self.assertEqual(record.knn_model_file.name, "sentiment_models/sentimen-test-v20/knn_model.joblib")
            self.assertEqual(record.svm_model_file.name, "sentiment_models/sentimen-test-v20/svm_model.joblib")
            self.assertEqual((media_dir / record.knn_model_file.name).read_bytes(), b"new-knn")
            self.assertEqual((media_dir / record.svm_model_file.name).read_bytes(), b"new-svm")
            history.refresh_from_db()
            prediction_history.refresh_from_db()
            self.assertEqual(history.model_version, new_version_name)
            self.assertEqual(prediction_history.model_version, new_version_name)

    def test_custom_admin_edit_model_overwrites_stable_filenames_without_duplicates(self):
        self.client.force_login(self.admin)
        version_name = "Sentimen Test V1.0"
        with tempfile.TemporaryDirectory() as temp_dir:
            media_dir = Path(temp_dir) / "media"
            with self.settings(SENTIMENT_MODELS_DIR=Path(temp_dir), MEDIA_ROOT=media_dir):
                record = SentimentModelVersion.objects.create(
                    version_name=version_name,
                    knn_model_file=SimpleUploadedFile("knn_model.joblib", b"old-knn"),
                    svm_model_file=SimpleUploadedFile("svm_linear_model.joblib", b"old-svm"),
                )

                first_knn_path = media_dir / record.knn_model_file.name
                first_svm_path = media_dir / record.svm_model_file.name

                response = self.client.post(
                    reverse("admin:model_edit", args=[version_name]),
                    {
                        "version_name": version_name,
                        "knn_model_file": SimpleUploadedFile("knn_model.joblib", b"new-knn"),
                        "svm_model_file": SimpleUploadedFile("svm_linear_model.joblib", b"new-svm"),
                    },
                )

            self.assertEqual(response.status_code, 302)
            record.refresh_from_db()
            self.assertEqual(record.knn_model_file.name, "sentiment_models/sentimen-test-v10/knn_model.joblib")
            self.assertEqual(record.svm_model_file.name, "sentiment_models/sentimen-test-v10/svm_model.joblib")
            self.assertEqual((media_dir / record.knn_model_file.name).read_bytes(), b"new-knn")
            self.assertEqual((media_dir / record.svm_model_file.name).read_bytes(), b"new-svm")
            self.assertTrue((media_dir / record.knn_model_file.name).exists())
            self.assertTrue((media_dir / record.svm_model_file.name).exists())
            self.assertFalse(first_knn_path.exists() and first_knn_path.name != "knn_model.joblib")
            self.assertFalse(first_svm_path.exists() and first_svm_path.name != "svm_model.joblib")
            model_files = sorted((media_dir / "sentiment_models" / "sentimen-test-v10").glob("*.joblib"))
            self.assertEqual([path.name for path in model_files], ["knn_model.joblib", "svm_model.joblib"])

    def test_custom_admin_edit_model_page_hides_delete_button(self):
        self.client.force_login(self.admin)
        version_name = "Sentimen Test V1.0"
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.settings(SENTIMENT_MODELS_DIR=Path(temp_dir), MEDIA_ROOT=Path(temp_dir) / "media"):
                SentimentModelVersion.objects.create(
                    version_name=version_name,
                    knn_model_file=SimpleUploadedFile("knn_model.joblib", b"old-knn"),
                    svm_model_file=SimpleUploadedFile("svm_linear_model.joblib", b"old-svm"),
                )
                response = self.client.get(reverse("admin:model_edit", args=[version_name]))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Hapus Model")

    def test_custom_admin_can_delete_model_version(self):
        self.client.force_login(self.admin)
        version_name = "Sentimen Test V1.0"
        with tempfile.TemporaryDirectory() as temp_dir:
            media_dir = Path(temp_dir) / "media"
            history = ScrapeHistory.objects.create(
                user=self.user,
                query="mobil listrik",
                model_version=version_name,
                language="in",
                start_date="2026-01-01",
                end_date="2026-01-02",
                tweet_count=1,
                rows=[],
            )
            prediction_history = PredictionHistory.objects.create(
                user=self.user,
                source_name="hasil.csv",
                model_version=version_name,
                text_column="text",
                sample_count=1,
                rows=[],
            )
            with self.settings(SENTIMENT_MODELS_DIR=Path(temp_dir), MEDIA_ROOT=media_dir):
                record = SentimentModelVersion.objects.create(
                    version_name=version_name,
                    knn_model_file=SimpleUploadedFile("knn_model.joblib", b"knn"),
                    svm_model_file=SimpleUploadedFile("svm_linear_model.joblib", b"svm"),
                )
                knn_path = media_dir / record.knn_model_file.name
                svm_path = media_dir / record.svm_model_file.name
                response = self.client.post(reverse("admin:model_delete", args=[version_name]))

            self.assertEqual(response.status_code, 302)
            self.assertFalse(SentimentModelVersion.objects.filter(version_name=version_name).exists())
            self.assertFalse(knn_path.exists())
            self.assertFalse(svm_path.exists())
            history.refresh_from_db()
            prediction_history.refresh_from_db()
            self.assertEqual(history.model_version, version_name)
            self.assertEqual(prediction_history.model_version, version_name)

    def test_inactive_user_cannot_login(self):
        self.user.is_active = False
        self.user.save(update_fields=["is_active"])

        response = self.client.post(
            reverse("login"),
            {
                "username": self.user.username,
                "password": "MemberPass123!",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_custom_admin_can_detail_edit_and_delete_history_datasets(self):
        scrape_history = ScrapeHistory.objects.create(
            user=self.other_user,
            query="query awal",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-02",
            tweet_count=1,
            rows=[{"id": "1", "text": "awal"}],
        )
        prediction_history = PredictionHistory.objects.create(
            user=self.other_user,
            source_name="awal.csv",
            text_column="text",
            sample_count=1,
            columns=["text"],
            rows=[{"text": "awal", "knn_label": "Positive", "svm_label": "Positive"}],
        )

        self.client.force_login(self.admin)
        scrape_detail = self.client.get(reverse("history_detail", args=[scrape_history.id]))
        prediction_detail = self.client.get(reverse("prediction_history_detail", args=[prediction_history.id]))
        self.assertEqual(scrape_detail.status_code, 200)
        self.assertEqual(prediction_detail.status_code, 200)

        prediction_edit = self.client.post(
            reverse("admin:prediction_history_edit", args=[prediction_history.id]),
            {
                "user": self.other_user.id,
                "source_name": "edited.csv",
                "text_column": "review",
                "sample_count": 2,
                "columns": '["review"]',
                "rows": '[{"review": "bagus", "knn_label": "Positive", "svm_label": "Positive"}]',
            },
        )
        self.assertEqual(prediction_edit.status_code, 302)
        prediction_history.refresh_from_db()
        self.assertEqual(prediction_history.source_name, "edited.csv")
        self.assertEqual(prediction_history.text_column, "review")
        self.assertEqual(prediction_history.rows[0]["review"], "bagus")

        scrape_edit = self.client.post(
            reverse("admin:scrape_history_edit", args=[scrape_history.id]),
            {
                "user": self.other_user.id,
                "query": "query edit",
                "language": "en",
                "start_date": "2026-02-01",
                "end_date": "2026-02-02",
                "tweet_count": 2,
                "rows": '[{"id": "2", "text": "edit"}]',
                "is_complete": "on",
                "resume_next_date": "",
                "stop_reason": "",
            },
        )
        self.assertEqual(scrape_edit.status_code, 302)
        scrape_history.refresh_from_db()
        self.assertEqual(scrape_history.query, "query edit")
        self.assertEqual(scrape_history.language, "en")
        self.assertEqual(scrape_history.rows[0]["text"], "edit")

        self.assertEqual(
            self.client.post(reverse("admin:prediction_history_delete", args=[prediction_history.id])).status_code,
            302,
        )
        self.assertEqual(
            self.client.post(reverse("admin:scrape_history_delete", args=[scrape_history.id])).status_code,
            302,
        )
        self.assertFalse(PredictionHistory.objects.filter(id=prediction_history.id).exists())
        self.assertFalse(ScrapeHistory.objects.filter(id=scrape_history.id).exists())

    def test_custom_admin_can_delete_selected_history_datasets(self):
        prediction_history = PredictionHistory.objects.create(
            user=self.other_user,
            source_name="dataset.csv",
            sample_count=3,
            rows=[],
        )
        scrape_history = ScrapeHistory.objects.create(
            user=self.other_user,
            query="dataset scraping",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-02",
            tweet_count=7,
            rows=[],
        )

        self.client.force_login(self.admin)

        prediction_delete = self.client.post(
            reverse("admin:history_delete_selected"),
            {
                "scope": "prediction",
                "selected_ids": [str(prediction_history.id)],
                "users_page": "1",
                "scrape_page": "1",
                "pred_page": "1",
                "dataset_tab": "prediction",
            },
        )
        self.assertEqual(prediction_delete.status_code, 302)
        self.assertFalse(PredictionHistory.objects.filter(id=prediction_history.id).exists())

        scrape_delete = self.client.post(
            reverse("admin:history_delete_selected"),
            {
                "scope": "scrape",
                "selected_ids": [str(scrape_history.id)],
                "users_page": "1",
                "scrape_page": "1",
                "pred_page": "1",
                "dataset_tab": "scraping",
            },
        )
        self.assertEqual(scrape_delete.status_code, 302)
        self.assertFalse(ScrapeHistory.objects.filter(id=scrape_history.id).exists())

    def test_pages_render_specific_browser_titles(self):
        login_response = self.client.get(reverse("login"))
        self.assertContains(login_response, "<title>Login | Sistem Analisis Sentimen</title>", html=True)

        scrape_history = ScrapeHistory.objects.create(
            user=self.user,
            query="judul query",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-02",
            tweet_count=1,
            rows=[{"id": "1", "text": "contoh"}],
        )
        prediction_history = PredictionHistory.objects.create(
            user=self.user,
            source_name="contoh.csv",
            text_column="text",
            sample_count=1,
            rows=[{"text": "contoh kalimat", "knn_label": "Positive", "svm_label": "Positive"}],
        )

        self.client.force_login(self.user)
        self.assertContains(
            self.client.get(reverse("home")),
            "<title>Beranda | Sistem Analisis Sentimen</title>",
            html=True,
        )
        home_response = self.client.get(reverse("home"))
        self.assertContains(home_response, '<span class="sidebar-logo-text">Sistem<br>Analisis Sentimen</span>', html=True)
        self.assertNotContains(home_response, "Membuat prediksi hasil klasifikasi sentimen.")
        self.assertContains(
            self.client.get(reverse("predict")),
            "<title>Buat Analisis | Sistem Analisis Sentimen</title>",
            html=True,
        )
        self.assertContains(
            self.client.get(reverse("history_list")),
            "<title>Riwayat Aktivitas | Sistem Analisis Sentimen</title>",
            html=True,
        )
        scrape_redirect_response = self.client.get(reverse("twitter_fetch"))
        self.assertEqual(scrape_redirect_response.status_code, 302)
        self.assertEqual(scrape_redirect_response.url, f"{reverse('predict')}?tab=scraping")
        self.assertContains(
            self.client.get(reverse("history_detail", args=[scrape_history.id])),
            "<title>Detail Riwayat Pengumpulan Data X | Sistem Analisis Sentimen</title>",
            html=True,
        )
        self.assertContains(
            self.client.get(reverse("prediction_history_detail", args=[prediction_history.id])),
            "<title>Detail Riwayat CSV/TXT | Sistem Analisis Sentimen</title>",
            html=True,
        )

        self.client.force_login(self.admin)
        self.assertContains(
            self.client.get(reverse("admin:index")),
            "<title>Admin Panel | Sistem Analisis Sentimen</title>",
            html=True,
        )
        self.assertContains(
            self.client.get(reverse("admin:user_add")),
            "<title>Tambah User | Sistem Analisis Sentimen</title>",
            html=True,
        )

    def test_home_shows_owner_scraping_and_prediction_totals(self):
        ScrapeHistory.objects.create(
            user=self.user,
            query="query_user_1",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-02",
            tweet_count=3,
            rows=[],
        )
        ScrapeHistory.objects.create(
            user=self.user,
            query="query_user_2",
            language="in",
            start_date="2026-01-03",
            end_date="2026-01-04",
            tweet_count=2,
            rows=[],
        )
        ScrapeHistory.objects.create(
            user=self.other_user,
            query="query_other",
            language="in",
            start_date="2026-01-05",
            end_date="2026-01-06",
            tweet_count=99,
            rows=[],
        )
        PredictionHistory.objects.create(
            user=self.user,
            sample_count=4,
            rows=[],
        )
        PredictionHistory.objects.create(
            user=self.other_user,
            sample_count=88,
            rows=[],
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("home"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Ringkasan Hasil")
        self.assertContains(response, "Hasil Scraping")
        self.assertContains(response, "Hasil CSV/TXT")
        self.assertContains(response, "Total Pengumpulan Data X")
        self.assertContains(response, "Total Tweet Terkumpul")
        self.assertContains(response, "Total File")
        self.assertContains(response, "Total Data")
        self.assertEqual(response.context["total_scraping_count"], 2)
        self.assertEqual(response.context["total_scraping_results"], 5)
        self.assertEqual(response.context["total_prediction_count"], 1)
        self.assertEqual(response.context["total_prediction_results"], 4)
        self.assertNotContains(response, "Website Ini Untuk Apa?")

    def test_scraping_post_creates_history_for_logged_user(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("twitter_fetch"),
            {
                "api_key": "dummy_api_key",
                "model_version": "Sentimen V1.0",
                "query": "mobil listrik",
                "language": "in",
                "start_date": "01/01/2026",
                "end_date": "02/01/2026",
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(ScrapeHistory.objects.count(), 1)
        history = ScrapeHistory.objects.get()
        self.assertEqual(history.user, self.user)
        self.assertEqual(history.model_version, "Sentimen V1.0")
        self.assertEqual(history.tweet_count, 0)
        self.assertFalse(history.is_complete)
        self.assertFalse(history.is_processing)
        self.assertEqual(str(history.resume_next_date), "2026-01-01")
        self.assertEqual(history.stop_reason, "processing")
        self.assertEqual(response.url, f"{reverse('history_detail', args=[history.id])}?auto=1")

    def test_predict_page_contains_scraping_tab_and_handles_scraping_submit(self):
        self.client.force_login(self.user)
        mocked_tweets = [
            {
                "id": "202",
                "text": "uji tab scraping",
                "CreatedAt": "2026-01-01T12:34:56+00:00",
            }
        ]
        mocked_predictions = [
            {
                "text": "uji tab scraping",
                "knn_label": "Positive",
                "knn_positive_score": 0.91,
                "knn_negative_score": 0.09,
                "svm_label": "Positive",
                "svm_positive_score": 0.88,
                "svm_negative_score": 0.12,
                "combined_label": "Positive",
                "combined_positive_score": 0.895,
                "combined_negative_score": 0.105,
            }
        ]

        page_response = self.client.get(reverse("predict") + "?tab=scraping")
        self.assertEqual(page_response.status_code, 200)
        self.assertContains(page_response, 'data-bs-target="#scraping-pane"')
        self.assertContains(page_response, 'class="predict-shared-model-bar"')
        self.assertContains(page_response, 'for="id_model_version">Versi Model</label>')
        self.assertContains(page_response, 'name="input_mode" value="scraping"')
        self.assertNotContains(page_response, reverse("twitter_fetch"))

        submit_response = self.client.post(
            reverse("predict"),
            {
                "input_mode": "scraping",
                "api_key": "dummy_api_key",
                "model_version": "Sentimen V1.0",
                "query": "mobil listrik",
                "language": "in",
                "start_date": "01/01/2026",
                "end_date": "02/01/2026",
            },
        )

        self.assertEqual(submit_response.status_code, 302)
        history = ScrapeHistory.objects.get()
        self.assertEqual(submit_response.url, f"{reverse('history_detail', args=[history.id])}?auto=1")

    def test_history_list_only_shows_owner_data(self):
        scrape_history = ScrapeHistory.objects.create(
            user=self.user,
            query="query_user",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-02",
            tweet_count=1,
            rows=[{"id": "1", "text": "a"}],
        )
        other_history = ScrapeHistory.objects.create(
            user=self.other_user,
            query="query_other",
            language="in",
            start_date="2026-01-03",
            end_date="2026-01-04",
            tweet_count=1,
            rows=[{"id": "2", "text": "b"}],
        )
        prediction_history = PredictionHistory.objects.create(
            user=self.user,
            source_name="dataset.csv",
            text_column="text",
            sample_count=2,
            rows=[{"text": "a"}, {"text": "b"}],
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("history_list"))
        self.assertContains(response, "query_user")
        self.assertContains(response, "Riwayat CSV/TXT")
        self.assertContains(response, 'class="history-card-grid"', count=2)
        self.assertContains(response, "1.")
        self.assertContains(response, "Dibuat")
        self.assertContains(response, "Jumlah Tweet")
        self.assertContains(response, "Kolom Teks")
        self.assertContains(response, "Jumlah Data")
        self.assertNotContains(response, "<table")
        self.assertContains(response, '<span class="badge text-bg-primary">CSV</span>', html=True)
        self.assertContains(response, "dataset.csv")
        self.assertContains(response, reverse("history_detail", args=[scrape_history.id]))
        self.assertContains(response, reverse("history_dashboard", args=[scrape_history.id]))
        self.assertContains(response, reverse("prediction_history_detail", args=[prediction_history.id]))
        self.assertContains(response, reverse("prediction_history_dashboard", args=[prediction_history.id]))
        self.assertNotContains(response, "prediksi user")
        self.assertContains(response, "Selesai", count=2)
        self.assertNotContains(response, "query_other")

        forbidden_detail = self.client.get(reverse("history_detail", args=[other_history.id]))
        self.assertEqual(forbidden_detail.status_code, 404)

    def test_history_list_hides_language_column_in_scrape_history(self):
        ScrapeHistory.objects.create(
            user=self.user,
            query="bahasa indonesia",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-02",
            tweet_count=1,
            rows=[],
        )
        ScrapeHistory.objects.create(
            user=self.user,
            query="english language",
            language="en",
            start_date="2026-01-03",
            end_date="2026-01-04",
            tweet_count=1,
            rows=[],
        )
        ScrapeHistory.objects.create(
            user=self.user,
            query="semua bahasa",
            language="",
            start_date="2026-01-05",
            end_date="2026-01-06",
            tweet_count=1,
            rows=[],
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("history_list"))

        self.assertNotContains(response, "<table")
        self.assertNotContains(response, "Bahasa Indonesia")
        self.assertNotContains(response, "Bahasa Inggris")
        self.assertNotContains(response, "Semua Bahasa")

    def test_scrape_history_detail_has_detail_header_and_back_button(self):
        history = ScrapeHistory.objects.create(
            user=self.user,
            query="mobil listrik",
            model_version="Sentimen V1.0",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-02",
            tweet_count=1,
            rows=[
                {
                    "id": "1",
                    "text": "mobil listrik bagus",
                    "CreatedAt": "2026-01-01T10:00:00+00:00",
                    "knn_label": "Positive",
                    "svm_label": "Positive",
                }
            ],
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("history_detail", args=[history.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Detail Riwayat Pengumpulan Data X")
        self.assertContains(response, ">Kembali</span>", html=False)
        self.assertContains(response, "Model: Sentimen V1.0")
        self.assertNotContains(response, "Status Riwayat Pengumpulan Data X")
        self.assertContains(response, "<th rowspan=\"2\">No.</th>", html=True)
        self.assertNotContains(response, "<th rowspan=\"2\">#</th>", html=True)
        self.assertNotContains(response, "Mulai Scraping")
        self.assertContains(response, reverse("history_dashboard", args=[history.id]))
        self.assertContains(response, "<span>Tabel</span>", html=False)
        self.assertContains(response, "<span>Dashboard</span>", html=False)
        self.assertNotContains(response, "Dashboard Hasil Scraping")

    def test_scrape_history_dashboard_view_shows_loading_state(self):
        history = ScrapeHistory.objects.create(
            user=self.user,
            query="mobil listrik",
            model_version="Sentimen V1.0",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-02",
            tweet_count=2,
            rows=[
                {
                    "id": "1",
                    "text": "mobil listrik bagus",
                    "CreatedAt": "2026-01-01T10:00:00+00:00",
                    "knn_label": "Positive",
                    "svm_label": "Positive",
                    "combined_label": "Positive",
                },
                {
                    "id": "2",
                    "text": "servis buruk",
                    "CreatedAt": "2026-01-02T11:00:00+00:00",
                    "knn_label": "Negative",
                    "svm_label": "Negative",
                    "combined_label": "Negative",
                },
            ],
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("history_dashboard", args=[history.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Dashboard Riwayat Pengumpulan Data X")
        self.assertContains(response, "Model: Sentimen V1.0")
        self.assertContains(response, "Memuat dashboard...")
        self.assertContains(response, reverse("history_dashboard_content", args=[history.id]))
        self.assertNotContains(response, "Dashboard Hasil Scraping")

    def test_scrape_history_detail_shows_processing_state(self):
        history = ScrapeHistory.objects.create(
            user=self.user,
            query="mobil listrik",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-02",
            tweet_count=0,
            rows=[],
            is_complete=False,
            is_processing=True,
            resume_next_date="2026-01-01",
            stop_reason="processing",
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("history_detail", args=[history.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Status Riwayat Pengumpulan Data X")
        self.assertContains(response, "Status scraping: <strong>Sedang Mencari..</strong>", html=True)
        self.assertContains(response, "Hasil tabel akan tampil otomatis setelah proses selesai.")
        self.assertNotContains(response, 'id="resumeScrapeForm"')

    def test_scrape_history_detail_hides_empty_warning_while_resume_is_pending(self):
        history = ScrapeHistory.objects.create(
            user=self.user,
            query="mobil listrik",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-02",
            tweet_count=0,
            rows=[],
            is_complete=False,
            is_processing=False,
            resume_next_date="2026-01-01",
            stop_reason="processing",
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("history_detail", args=[history.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Status Riwayat Pengumpulan Data X")
        self.assertNotContains(response, "Data riwayat scraping kosong.")

    def test_scrape_history_dashboard_content_returns_dashboard_html(self):
        history = ScrapeHistory.objects.create(
            user=self.user,
            query="mobil listrik",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-02",
            tweet_count=2,
            rows=[
                {
                    "id": "1",
                    "text": "mobil listrik bagus",
                    "CreatedAt": "2026-01-01T10:00:00+00:00",
                    "knn_label": "Positive",
                    "svm_label": "Positive",
                    "combined_label": "Positive",
                },
                {
                    "id": "2",
                    "text": "servis buruk",
                    "CreatedAt": "2026-01-02T11:00:00+00:00",
                    "knn_label": "Negative",
                    "svm_label": "Negative",
                    "combined_label": "Negative",
                },
            ],
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("history_dashboard_content", args=[history.id]))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("Rentang data: 01 Januari 2026 - 02 Januari 2026", payload["html"])
        self.assertIn("Sentimen Positif: 1", payload["html"])
        self.assertIn("Unduh", payload["html"])
        self.assertIn("data-dashboard-download", payload["html"])
        self.assertIn("scraping-dashboard-data", payload["html"])

    def test_predict_file_creates_prediction_history(self):
        self.client.force_login(self.user)
        uploaded = SimpleUploadedFile("uji.csv", b"text\nmobil listrik bagus\n")
        mocked_texts = ["mobil listrik bagus", "servis buruk"]
        mocked_source_rows = [
            {"text": "mobil listrik bagus"},
            {"text": "servis buruk"},
        ]
        mocked_source_columns = ["text"]
        mocked_predictions = [
            {
                "knn_label": "Positive",
                "knn_positive_score": 0.91,
                "knn_negative_score": 0.09,
                "svm_label": "Positive",
                "svm_positive_score": 0.87,
                "svm_negative_score": 0.13,
                "combined_label": "Positive",
                "combined_positive_score": 0.89,
                "combined_negative_score": 0.11,
            },
            {
                "knn_label": "Negative",
                "knn_positive_score": 0.16,
                "knn_negative_score": 0.84,
                "svm_label": "Negative",
                "svm_positive_score": 0.12,
                "svm_negative_score": 0.88,
                "combined_label": "Negative",
                "combined_positive_score": 0.14,
                "combined_negative_score": 0.86,
            },
        ]

        with patch(
            "sentiment_app.views._launch_background_history_job",
            side_effect=self._run_background_job_inline,
        ), patch(
            "sentiment_app.views.parse_uploaded_file",
            return_value=(mocked_texts, "text", mocked_source_rows, mocked_source_columns),
        ), patch(
            "sentiment_app.views.predict_batch",
            return_value=mocked_predictions,
        ) as mocked_predict:
            response = self.client.post(
                reverse("predict"),
                {
                    "input_mode": "file",
                    "model_version": "Sentimen V1.0",
                    "text_column": "",
                    "upload_file": uploaded,
                },
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(PredictionHistory.objects.count(), 1)
        history = PredictionHistory.objects.get()
        self.assertEqual(response.url, reverse("prediction_history_detail", args=[history.id]))
        self.assertEqual(history.user, self.user)
        self.assertEqual(mocked_predict.call_args.kwargs.get("model_version"), "Sentimen V1.0")
        self.assertEqual(history.source_name, "uji.csv")
        self.assertEqual(history.model_version, "Sentimen V1.0")
        self.assertFalse(history.is_processing)
        self.assertEqual(history.sample_count, 2)
        self.assertEqual(history.rows[0]["text"], "mobil listrik bagus")
        self.assertEqual(history.rows[1]["svm_label"], "Negative")
        self.assertAlmostEqual(history.rows[0]["svm_positive_score"], 0.87, places=4)
        self.assertAlmostEqual(history.rows[1]["svm_negative_score"], 0.88, places=4)
        self.assertEqual(history.rows[0]["combined_label"], "Positive")
        self.assertAlmostEqual(history.rows[1]["combined_positive_score"], 0.14, places=4)
        self.assertAlmostEqual(history.rows[1]["combined_negative_score"], 0.86, places=4)

    @override_settings(SENTIMENT_WORDCLOUD_MAX_ROWS=1)
    def test_prediction_file_history_detail_shows_dashboard(self):
        history = PredictionHistory.objects.create(
            user=self.user,
            source_name="uji.csv",
            model_version="Sentimen V1.0",
            text_column="review",
            sample_count=2,
            columns=["review", "tanggal"],
            rows=[
                {
                    "review": "mobil listrik bagus",
                    "tanggal": "2026-01-01",
                    "knn_label": "Positive",
                    "knn_positive_score": 0.91,
                    "knn_negative_score": 0.09,
                    "svm_label": "Positive",
                    "svm_positive_score": 0.87,
                    "svm_negative_score": 0.13,
                },
                {
                    "review": "servis buruk",
                    "tanggal": "2026-01-02",
                    "knn_label": "Negative",
                    "knn_positive_score": 0.16,
                    "knn_negative_score": 0.84,
                    "svm_label": "Negative",
                    "svm_positive_score": 0.12,
                    "svm_negative_score": 0.88,
                },
            ],
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("prediction_history_dashboard", args=[history.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Dashboard Riwayat CSV/TXT")
        self.assertContains(response, "Model: Sentimen V1.0")
        self.assertContains(response, "Memuat dashboard...")
        self.assertContains(response, reverse("prediction_history_dashboard_content", args=[history.id]))
        self.assertNotContains(response, "Dashboard Hasil Prediksi")

    @override_settings(SENTIMENT_WORDCLOUD_MAX_ROWS=1)
    def test_prediction_history_dashboard_content_returns_dashboard_html(self):
        history = PredictionHistory.objects.create(
            user=self.user,
            source_name="uji.csv",
            text_column="review",
            sample_count=2,
            columns=["review", "tanggal"],
            rows=[
                {
                    "review": "mobil listrik bagus",
                    "tanggal": "2026-01-01",
                    "knn_label": "Positive",
                    "knn_positive_score": 0.91,
                    "knn_negative_score": 0.09,
                    "svm_label": "Positive",
                    "svm_positive_score": 0.87,
                    "svm_negative_score": 0.13,
                },
                {
                    "review": "servis buruk",
                    "tanggal": "2026-01-02",
                    "knn_label": "Negative",
                    "knn_positive_score": 0.16,
                    "knn_negative_score": 0.84,
                    "svm_label": "Negative",
                    "svm_positive_score": 0.12,
                    "svm_negative_score": 0.88,
                },
            ],
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("prediction_history_dashboard_content", args=[history.id]))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("Rentang data:", payload["html"])
        self.assertIn("Sentimen Positif: 1", payload["html"])
        self.assertIn("Unduh", payload["html"])
        self.assertIn("data-dashboard-download", payload["html"])
        self.assertIn("prediction-dashboard-data", payload["html"])

    def test_prediction_file_history_detail_hides_dashboard_by_default(self):
        history = PredictionHistory.objects.create(
            user=self.user,
            source_name="uji.csv",
            model_version="Sentimen V1.0",
            text_column="review",
            sample_count=1,
            columns=["review"],
            rows=[
                {
                    "review": "mobil listrik bagus",
                    "knn_label": "Positive",
                    "knn_positive_score": 0.91,
                    "knn_negative_score": 0.09,
                    "svm_label": "Positive",
                    "svm_positive_score": 0.87,
                    "svm_negative_score": 0.13,
                }
            ],
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("prediction_history_detail", args=[history.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, reverse("prediction_history_dashboard", args=[history.id]))
        self.assertContains(response, "Model: Sentimen V1.0")
        self.assertNotContains(response, "Status Riwayat CSV/TXT")
        self.assertNotContains(response, "Dashboard Hasil Prediksi")

    def test_prediction_file_history_detail_shows_processing_state(self):
        history = PredictionHistory.objects.create(
            user=self.user,
            source_name="uji.csv",
            text_column="review",
            sample_count=0,
            columns=[],
            rows=[],
            is_processing=True,
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("prediction_history_detail", args=[history.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Status Riwayat CSV/TXT")
        self.assertContains(response, "Status: Ongoing")
        self.assertContains(response, "Hasil CSV/TXT akan tampil otomatis setelah selesai.")

    def test_prediction_file_history_detail_hides_id_column_in_preview(self):
        history = PredictionHistory.objects.create(
            user=self.user,
            source_name="uji.csv",
            text_column="review",
            sample_count=1,
            columns=["id", "review"],
            rows=[
                {
                    "id": "abc-001",
                    "review": "mobil listrik bagus",
                    "knn_label": "Positive",
                    "knn_positive_score": 0.91,
                    "knn_negative_score": 0.09,
                    "svm_label": "Positive",
                    "svm_positive_score": 0.87,
                    "svm_negative_score": 0.13,
                }
            ],
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("prediction_history_detail", args=[history.id]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(history.columns[0], "id")
        self.assertEqual(history.rows[0]["id"], "abc-001")
        self.assertNotIn("id", response.context["batch_preview_headers"])
        self.assertContains(response, "Nilai Probabilitas (Skor 0-1)")
        self.assertContains(response, "Positif (KNN)")
        self.assertContains(response, "Negatif (Soft Voting)")
        self.assertContains(response, "Unduh")
        self.assertContains(response, reverse("download_prediction_history_csv", args=[history.id]))
        self.assertContains(response, reverse("download_prediction_history_xlsx", args=[history.id]))
        self.assertContains(response, "Excel (.xlsx)")

    def test_prediction_history_detail_is_user_scoped(self):
        own_history = PredictionHistory.objects.create(
            user=self.user,
            source_name="own.csv",
            text_column="text",
            sample_count=1,
            rows=[{"text": "tes", "knn_label": "Positive", "svm_label": "Positive"}],
        )
        other_history = PredictionHistory.objects.create(
            user=self.other_user,
            source_name="other.csv",
            text_column="text",
            sample_count=1,
            rows=[{"text": "other", "knn_label": "Negative", "svm_label": "Negative"}],
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("prediction_history_detail", args=[own_history.id]))
        self.assertEqual(response.status_code, 200)
        forbidden_detail = self.client.get(reverse("prediction_history_detail", args=[other_history.id]))
        self.assertEqual(forbidden_detail.status_code, 404)

    def test_prediction_history_csv_download_uses_flat_headers_and_keeps_source_id(self):
        history = PredictionHistory.objects.create(
            user=self.user,
            source_name="uji.csv",
            text_column="review",
            sample_count=1,
            columns=["id", "review"],
            rows=[
                {
                    "id": "abc-001",
                    "review": "mobil listrik bagus",
                    "knn_label": "Positive",
                    "knn_positive_score": 0.91,
                    "knn_negative_score": 0.09,
                    "svm_label": "Positive",
                    "svm_positive_score": 0.87,
                    "svm_negative_score": 0.13,
                    "combined_label": "Positive",
                    "combined_positive_score": 0.89,
                    "combined_negative_score": 0.11,
                }
            ],
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("download_prediction_history_csv", args=[history.id]))
        csv_text = response.content.decode("utf-8")
        header_row = csv_text.splitlines()[0]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv; charset=utf-8")
        self.assertIn("attachment; filename=\"uji.csv\"", response["Content-Disposition"])
        self.assertIn("id", header_row)
        self.assertIn("review", header_row)
        self.assertIn("Probabilitas Positif KNN", header_row)
        self.assertIn("Probabilitas Negatif Soft Voting", header_row)
        self.assertIn("Soft Voting", header_row)
        self.assertNotIn("Skor 0-1", csv_text)
        self.assertIn("abc-001", csv_text)
        self.assertIn("0.910000", csv_text)
        self.assertIn("Positif", csv_text)
        self.assertNotIn("Positive", csv_text)

    def test_scrape_history_csv_download_uses_flat_headers(self):
        history = ScrapeHistory.objects.create(
            user=self.user,
            query="dataset scraping",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-02",
            tweet_count=1,
            rows=[
                {
                    "id": "1",
                    "url": "https://x.com/status/1",
                    "text": "mobil listrik bagus",
                    "retweetCount": 4,
                    "replyCount": 2,
                    "likeCount": 9,
                    "quoteCount": 1,
                    "viewCount": 30,
                    "CreatedAt": "2026-01-01T10:00:00+00:00",
                    "lang": "in",
                    "bookmarkCount": 0,
                    "isReply": False,
                    "inReplyTold": "",
                    "userName": "akun_uji",
                    "image_tweet": "https://example.com/image.jpg",
                    "knn_label": "Positive",
                    "knn_positive_score": 0.91,
                    "knn_negative_score": 0.09,
                    "svm_label": "Positive",
                    "svm_positive_score": 0.87,
                    "svm_negative_score": 0.13,
                    "combined_label": "Positive",
                    "combined_positive_score": 0.89,
                    "combined_negative_score": 0.11,
                }
            ],
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("download_scrape_history_csv", args=[history.id]))
        csv_text = response.content.decode("utf-8")
        header_row = csv_text.splitlines()[0]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv; charset=utf-8")
        self.assertIn("attachment; filename=\"scrape_history_{}.csv\"".format(history.id), response["Content-Disposition"])
        self.assertIn("ID", header_row)
        self.assertIn("Teks", header_row)
        self.assertIn("Probabilitas Positif KNN", header_row)
        self.assertIn("Probabilitas Negatif Soft Voting", header_row)
        self.assertIn("KNN", header_row)
        self.assertIn("Soft Voting", header_row)
        self.assertNotIn("Skor 0-1", csv_text)
        self.assertIn("mobil listrik bagus", csv_text)
        self.assertIn("0.890000", csv_text)
        self.assertIn("Positif", csv_text)
        self.assertNotIn("Positive", csv_text)

    def test_prediction_history_xlsx_download_uses_flat_headers_and_keeps_source_id(self):
        history = PredictionHistory.objects.create(
            user=self.user,
            source_name="uji.csv",
            text_column="review",
            sample_count=1,
            columns=["id", "review"],
            rows=[
                {
                    "id": "abc-001",
                    "review": "mobil listrik bagus",
                    "knn_label": "Positive",
                    "knn_positive_score": 0.91,
                    "knn_negative_score": 0.09,
                    "svm_label": "Positive",
                    "svm_positive_score": 0.87,
                    "svm_negative_score": 0.13,
                    "combined_label": "Positive",
                    "combined_positive_score": 0.89,
                    "combined_negative_score": 0.11,
                }
            ],
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("download_prediction_history_xlsx", args=[history.id]))
        workbook = zipfile.ZipFile(io.BytesIO(response.content))
        sheet_xml = workbook.read("xl/worksheets/sheet1.xml").decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response["Content-Type"],
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        self.assertIn("attachment; filename=\"uji.xlsx\"", response["Content-Disposition"])
        self.assertIn("id", sheet_xml)
        self.assertIn("review", sheet_xml)
        self.assertIn("Probabilitas Positif KNN", sheet_xml)
        self.assertIn("Probabilitas Negatif Soft Voting", sheet_xml)
        self.assertIn("abc-001", sheet_xml)
        self.assertIn("0.910000", sheet_xml)
        self.assertIn("Positif", sheet_xml)

    def test_scrape_history_xlsx_download_uses_flat_headers(self):
        history = ScrapeHistory.objects.create(
            user=self.user,
            query="dataset scraping",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-02",
            tweet_count=1,
            rows=[
                {
                    "id": "1",
                    "url": "https://x.com/status/1",
                    "text": "mobil listrik bagus",
                    "retweetCount": 4,
                    "replyCount": 2,
                    "likeCount": 9,
                    "quoteCount": 1,
                    "viewCount": 30,
                    "CreatedAt": "2026-01-01T10:00:00+00:00",
                    "lang": "in",
                    "bookmarkCount": 0,
                    "isReply": False,
                    "inReplyTold": "",
                    "userName": "akun_uji",
                    "image_tweet": "https://example.com/image.jpg",
                    "knn_label": "Positive",
                    "knn_positive_score": 0.91,
                    "knn_negative_score": 0.09,
                    "svm_label": "Positive",
                    "svm_positive_score": 0.87,
                    "svm_negative_score": 0.13,
                    "combined_label": "Positive",
                    "combined_positive_score": 0.89,
                    "combined_negative_score": 0.11,
                }
            ],
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("download_scrape_history_xlsx", args=[history.id]))
        workbook = zipfile.ZipFile(io.BytesIO(response.content))
        sheet_xml = workbook.read("xl/worksheets/sheet1.xml").decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response["Content-Type"],
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        self.assertIn(
            "attachment; filename=\"scrape_history_{}.xlsx\"".format(history.id),
            response["Content-Disposition"],
        )
        self.assertIn("ID", sheet_xml)
        self.assertIn("Teks", sheet_xml)
        self.assertIn("Probabilitas Positif KNN", sheet_xml)
        self.assertIn("Probabilitas Negatif Soft Voting", sheet_xml)
        self.assertIn("mobil listrik bagus", sheet_xml)
        self.assertIn("0.890000", sheet_xml)
        self.assertIn("Positif", sheet_xml)

    def test_prediction_history_detail_upgrades_legacy_svm_scores(self):
        history = PredictionHistory.objects.create(
            user=self.user,
            source_name="legacy.csv",
            text_column="review",
            sample_count=1,
            rows=[
                {
                    "review": "mobil listrik bagus",
                    "knn_label": "Positive",
                    "knn_score": 0.91,
                    "svm_label": "Positive",
                    "svm_score": 0.7,
                }
            ],
        )

        self.client.force_login(self.user)
        with patch(
            "sentiment_app.views.predict_batch",
            return_value=[
                {
                    "text": "mobil listrik bagus",
                    "knn_label": "Positive",
                    "knn_positive_score": 0.91,
                    "knn_negative_score": 0.09,
                    "svm_label": "Positive",
                    "svm_positive_score": 0.7311,
                    "svm_negative_score": 0.2689,
                    "combined_label": "Positive",
                    "combined_positive_score": 0.8205,
                    "combined_negative_score": 0.1795,
                }
            ],
        ):
            response = self.client.get(reverse("prediction_history_detail", args=[history.id]))

        history.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(history.rows[0]["svm_label"], "Positive")
        self.assertAlmostEqual(history.rows[0]["svm_positive_score"], 0.7311, places=4)
        self.assertAlmostEqual(history.rows[0]["svm_negative_score"], 0.2689, places=4)
        self.assertNotIn("svm_score", history.rows[0])
        self.assertEqual(history.rows[0]["combined_label"], "Positive")
        self.assertAlmostEqual(history.rows[0]["combined_positive_score"], 0.8205, places=4)
        self.assertAlmostEqual(history.rows[0]["combined_negative_score"], 0.1795, places=4)
        self.assertContains(response, "0,7311")

    def test_prediction_history_detail_reupgrades_version_two_history(self):
        history = PredictionHistory.objects.create(
            user=self.user,
            source_name="version2.csv",
            text_column="review",
            sample_count=1,
            rows=[
                {
                    "review": "teks netral",
                    "knn_label": "Neutral",
                    "knn_score": 0.5,
                    "svm_label": "Positive",
                    "svm_score": 0.52,
                }
            ],
        )

        self.client.force_login(self.user)
        with patch(
            "sentiment_app.views.predict_batch",
            return_value=[
                {
                    "text": "teks netral",
                    "knn_label": "Neutral",
                    "knn_positive_score": 0.5,
                    "knn_negative_score": 0.5,
                    "svm_label": "Neutral",
                    "svm_positive_score": 0.5,
                    "svm_negative_score": 0.5,
                    "combined_label": "Neutral",
                    "combined_positive_score": 0.5,
                    "combined_negative_score": 0.5,
                }
            ],
        ):
            response = self.client.get(reverse("prediction_history_detail", args=[history.id]))

        history.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(history.rows[0]["svm_label"], "Neutral")
        self.assertAlmostEqual(history.rows[0]["svm_positive_score"], 0.5, places=4)
        self.assertAlmostEqual(history.rows[0]["svm_negative_score"], 0.5, places=4)
        self.assertEqual(history.rows[0]["combined_label"], "Neutral")
        self.assertAlmostEqual(history.rows[0]["combined_positive_score"], 0.5, places=4)
        self.assertAlmostEqual(history.rows[0]["combined_negative_score"], 0.5, places=4)

    def test_scrape_history_detail_upgrades_legacy_svm_scores_in_chunks(self):
        history = ScrapeHistory.objects.create(
            user=self.user,
            query="dataset scraping",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-02",
            tweet_count=2,
            rows=[
                {
                    "id": "1",
                    "text": "mobil listrik bagus",
                    "knn_label": "Positive",
                    "knn_score": 0.91,
                    "svm_label": "Positive",
                    "svm_score": 0.7,
                }
            ],
        )
        chunk = ScrapeTempChunk.objects.create(
            history=history,
            chunk_index=0,
            rows=[
                {
                    "id": "2",
                    "text": "servis buruk",
                    "knn_label": "Negative",
                    "knn_score": 0.12,
                    "svm_label": "Negative",
                    "svm_score": -0.4,
                }
            ],
        )

        self.client.force_login(self.user)
        with patch(
            "sentiment_app.views.predict_batch",
            return_value=[
                {
                    "text": "mobil listrik bagus",
                    "knn_label": "Positive",
                    "knn_positive_score": 0.91,
                    "knn_negative_score": 0.09,
                    "svm_label": "Positive",
                    "svm_positive_score": 0.7311,
                    "svm_negative_score": 0.2689,
                    "combined_label": "Positive",
                    "combined_positive_score": 0.8205,
                    "combined_negative_score": 0.1795,
                },
                {
                    "text": "servis buruk",
                    "knn_label": "Negative",
                    "knn_positive_score": 0.12,
                    "knn_negative_score": 0.88,
                    "svm_label": "Negative",
                    "svm_positive_score": 0.4013,
                    "svm_negative_score": 0.5987,
                    "combined_label": "Negative",
                    "combined_positive_score": 0.2607,
                    "combined_negative_score": 0.7393,
                },
            ],
        ):
            response = self.client.get(reverse("history_detail", args=[history.id]))

        history.refresh_from_db()
        chunk.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertAlmostEqual(history.rows[0]["svm_positive_score"], 0.7311, places=4)
        self.assertAlmostEqual(history.rows[0]["svm_negative_score"], 0.2689, places=4)
        self.assertAlmostEqual(chunk.rows[0]["svm_positive_score"], 0.4013, places=4)
        self.assertAlmostEqual(chunk.rows[0]["svm_negative_score"], 0.5987, places=4)
        self.assertAlmostEqual(history.rows[0]["combined_positive_score"], 0.8205, places=4)
        self.assertAlmostEqual(history.rows[0]["combined_negative_score"], 0.1795, places=4)
        self.assertAlmostEqual(chunk.rows[0]["combined_positive_score"], 0.2607, places=4)
        self.assertAlmostEqual(chunk.rows[0]["combined_negative_score"], 0.7393, places=4)

    @override_settings(SENTIMENT_TWITTER_TEMP_DB_THRESHOLD_DAYS=30)
    def test_scraping_long_range_uses_temp_db_chunks(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("twitter_fetch"),
            {
                "api_key": "dummy_api_key",
                "model_version": "Sentimen V1.0",
                "query": "mobil listrik",
                "language": "in",
                "start_date": "01/01/2026",
                "end_date": "05/03/2026",
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(ScrapeHistory.objects.count(), 1)
        history = ScrapeHistory.objects.get()
        self.assertEqual(history.model_version, "Sentimen V1.0")
        self.assertEqual(history.tweet_count, 0)
        self.assertFalse(history.is_complete)
        self.assertFalse(history.is_processing)
        self.assertEqual(history.rows, [])
        self.assertEqual(response.url, f"{reverse('history_detail', args=[history.id])}?auto=1")

    def test_two_week_scraping_uses_daily_windows(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("twitter_fetch"),
            {
                "api_key": "dummy_api_key",
                "model_version": "Sentimen V1.0",
                "query": "mobil listrik",
                "language": "in",
                "start_date": "01/01/2026",
                "end_date": "14/01/2026",
            },
        )

        self.assertEqual(response.status_code, 302)
        history = ScrapeHistory.objects.get()
        self.assertEqual(str(history.resume_next_date), "2026-01-01")
        self.assertEqual(response.url, f"{reverse('history_detail', args=[history.id])}?auto=1")

    def test_scraping_marks_history_incomplete_when_partial_timeout(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("twitter_fetch"),
            {
                "api_key": "dummy_api_key",
                "model_version": "Sentimen V1.0",
                "query": "mobil listrik",
                "language": "in",
                "start_date": "01/01/2026",
                "end_date": "03/01/2026",
            },
        )

        self.assertEqual(response.status_code, 302)
        history = ScrapeHistory.objects.get()
        self.assertEqual(history.model_version, "Sentimen V1.0")
        self.assertFalse(history.is_processing)
        self.assertFalse(history.is_complete)
        self.assertEqual(str(history.resume_next_date), "2026-01-01")
        self.assertEqual(history.stop_reason, "processing")
        self.assertEqual(response.url, f"{reverse('history_detail', args=[history.id])}?auto=1")

    def test_resume_scrape_appends_rows_and_can_complete(self):
        self.client.force_login(self.user)
        history = ScrapeHistory.objects.create(
            user=self.user,
            query="mobil listrik",
            model_version="Sentimen V1.0",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-03",
            tweet_count=1,
            rows=[
                {
                    "id": "seed-1",
                    "text": "data awal",
                    "CreatedAt": "2026-01-01T10:00:00+00:00",
                    "knn_label": "Positive",
                    "svm_label": "Positive",
                }
            ],
            is_complete=False,
            resume_next_date="2026-01-02",
            stop_reason="timed_out",
        )

        window_rows = [
            {
                "id": "seed-2",
                "text": "data lanjutan",
                "CreatedAt": "2026-01-02T11:00:00+00:00",
            }
        ]
        window_predictions = [
            {
                "text": "data lanjutan",
                "knn_label": "Negative",
                "knn_positive_score": 0.2,
                "knn_negative_score": 0.8,
                "svm_label": "Negative",
                "svm_positive_score": 0.3,
                "svm_negative_score": 0.7,
                "combined_label": "Negative",
                "combined_positive_score": 0.25,
                "combined_negative_score": 0.75,
            }
        ]

        def _fake_fetch_tweets(*args, **kwargs):
            callback = kwargs.get("on_window")
            if callback:
                callback(window_rows)
            return [], {"next_start_date": "2026-01-04", "rate_limited": False, "timed_out": False, "truncated": False}

        with patch("sentiment_app.views.fetch_tweets", side_effect=_fake_fetch_tweets), patch(
            "sentiment_app.views.predict_batch_in_chunks",
            return_value=window_predictions,
        ) as mocked_predict:
            response = self.client.post(
                reverse("resume_scrape", args=[history.id]),
                {
                    "api_key": "dummy_api_key",
                    "per_page": 10,
                },
            )

        self.assertEqual(response.status_code, 302)
        history.refresh_from_db()
        self.assertEqual(mocked_predict.call_args.kwargs.get("model_version"), "Sentimen V1.0")
        self.assertTrue(history.is_complete)
        self.assertIsNone(history.resume_next_date)
        self.assertEqual(history.tweet_count, 2)
        self.assertEqual(ScrapeTempChunk.objects.filter(history=history).count(), 1)

    def test_resume_scrape_only_advances_one_day_per_request(self):
        self.client.force_login(self.user)
        history = ScrapeHistory.objects.create(
            user=self.user,
            query="mobil listrik",
            model_version="Sentimen V1.0",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-03",
            tweet_count=0,
            rows=[],
            is_complete=False,
            resume_next_date="2026-01-01",
            stop_reason="processing",
        )

        window_rows = [
            {
                "id": "seed-10",
                "text": "data hari pertama",
                "CreatedAt": "2026-01-01T11:00:00+00:00",
            }
        ]
        window_predictions = [
            {
                "text": "data hari pertama",
                "knn_label": "Positive",
                "knn_positive_score": 0.7,
                "knn_negative_score": 0.3,
                "svm_label": "Positive",
                "svm_positive_score": 0.6,
                "svm_negative_score": 0.4,
                "combined_label": "Positive",
                "combined_positive_score": 0.65,
                "combined_negative_score": 0.35,
            }
        ]

        def _fake_fetch_tweets(*args, **kwargs):
            callback = kwargs.get("on_window")
            if callback:
                callback(window_rows)
            return [], {"next_start_date": "2026-01-02", "rate_limited": False, "timed_out": False, "truncated": False}

        with patch("sentiment_app.views.fetch_tweets", side_effect=_fake_fetch_tweets), patch(
            "sentiment_app.views.predict_batch_in_chunks",
            return_value=window_predictions,
        ):
            response = self.client.post(
                reverse("resume_scrape", args=[history.id]),
                {
                    "api_key": "dummy_api_key",
                    "per_page": 10,
                },
            )

        self.assertEqual(response.status_code, 302)
        history.refresh_from_db()
        self.assertFalse(history.is_complete)
        self.assertEqual(str(history.resume_next_date), "2026-01-02")
        self.assertEqual(history.tweet_count, 1)

    def test_resume_scrape_ajax_returns_progress_payload(self):
        self.client.force_login(self.user)
        history = ScrapeHistory.objects.create(
            user=self.user,
            query="mobil listrik",
            model_version="Sentimen V1.0",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-03",
            tweet_count=0,
            rows=[],
            is_complete=False,
            resume_next_date="2026-01-02",
            stop_reason="timed_out",
        )

        window_rows = [
            {
                "id": "seed-9",
                "text": "data lanjutan ajax",
                "CreatedAt": "2026-01-02T11:00:00+00:00",
            }
        ]
        window_predictions = [
            {
                "text": "data lanjutan ajax",
                "knn_label": "Positive",
                "knn_positive_score": 0.7,
                "knn_negative_score": 0.3,
                "svm_label": "Positive",
                "svm_positive_score": 0.6,
                "svm_negative_score": 0.4,
                "combined_label": "Positive",
                "combined_positive_score": 0.65,
                "combined_negative_score": 0.35,
            }
        ]

        def _fake_fetch_tweets(*args, **kwargs):
            callback = kwargs.get("on_window")
            if callback:
                callback(window_rows)
            return [], {"next_start_date": "2026-01-03", "rate_limited": False, "timed_out": True, "truncated": False}

        with patch("sentiment_app.views.fetch_tweets", side_effect=_fake_fetch_tweets), patch(
            "sentiment_app.views.predict_batch_in_chunks",
            return_value=window_predictions,
        ) as mocked_predict:
            response = self.client.post(
                reverse("resume_scrape", args=[history.id]),
                {
                    "api_key": "dummy_api_key",
                    "ajax": "1",
                },
                HTTP_X_REQUESTED_WITH="XMLHttpRequest",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mocked_predict.call_args.kwargs.get("model_version"), "Sentimen V1.0")
        payload = response.json()
        self.assertTrue(payload.get("ok"))
        self.assertIn("progress_pct", payload)
        self.assertIn("tweet_count", payload)

    def test_resume_scrape_ajax_rate_limit_returns_retryable_payload(self):
        self.client.force_login(self.user)
        history = ScrapeHistory.objects.create(
            user=self.user,
            query="mobil listrik",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-15",
            tweet_count=0,
            rows=[],
            is_complete=False,
            resume_next_date="2026-01-02",
            stop_reason="rate_limited",
        )

        with patch(
            "sentiment_app.views.fetch_tweets",
            side_effect=TwitterRateLimitError("Batas permintaan tercapai. Coba lagi dalam beberapa menit."),
        ):
            response = self.client.post(
                reverse("resume_scrape", args=[history.id]),
                {
                    "api_key": "dummy_api_key",
                    "ajax": "1",
                },
                HTTP_X_REQUESTED_WITH="XMLHttpRequest",
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload.get("ok"))
        self.assertTrue(payload.get("retryable"))
        self.assertEqual(payload.get("error_code"), "rate_limited")
        self.assertEqual(payload.get("retry_after_seconds"), 8)

    def test_resume_scrape_ajax_timeout_returns_retryable_payload(self):
        self.client.force_login(self.user)
        history = ScrapeHistory.objects.create(
            user=self.user,
            query="mobil listrik",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-15",
            tweet_count=0,
            rows=[],
            is_complete=False,
            resume_next_date="2026-01-02",
            stop_reason="timed_out",
        )

        with patch(
            "sentiment_app.views.fetch_tweets",
            side_effect=TwitterTimeoutError("Proses scraping melebihi batas waktu server."),
        ):
            response = self.client.post(
                reverse("resume_scrape", args=[history.id]),
                {
                    "api_key": "dummy_api_key",
                    "ajax": "1",
                },
                HTTP_X_REQUESTED_WITH="XMLHttpRequest",
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload.get("ok"))
        self.assertTrue(payload.get("retryable"))
        self.assertEqual(payload.get("error_code"), "timed_out")
        self.assertEqual(payload.get("retry_after_seconds"), 3)
