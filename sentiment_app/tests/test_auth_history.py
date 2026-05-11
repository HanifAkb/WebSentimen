import tempfile
from pathlib import Path
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from sentiment_app.models import PredictionHistory, ScrapeHistory, ScrapeTempChunk
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
        ScrapeHistory.objects.create(
            user=self.other_user,
            query="dataset scraping",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-02",
            tweet_count=7,
            rows=[],
        )
        PredictionHistory.objects.create(
            user=self.other_user,
            input_type=PredictionHistory.InputType.FILE,
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
        self.assertContains(response, "Dataset PredictionHistory")
        self.assertContains(response, "Dataset ScrapeHistory")
        self.assertContains(response, "<th>No.</th>", html=True)
        self.assertContains(response, "<th>ID</th>", html=True)
        self.assertContains(response, "Nama Lengkap")
        self.assertContains(response, "member")
        self.assertContains(response, "dataset scraping")
        self.assertContains(response, "dataset.csv")

    def test_custom_admin_can_create_edit_and_delete_user(self):
        self.client.force_login(self.admin)
        create_response = self.client.post(
            reverse("admin:user_add"),
            {
                "username": "created_user",
                "full_name": "Created User",
                "email": "created@example.com",
                "is_staff": "on",
                "is_superuser": "",
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
                "is_staff": "",
                "is_superuser": "on",
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
        self.assertNotContains(response, "Hapus User")
        self.assertNotContains(response, reverse("admin:user_delete", args=[self.user.id]))

    def test_custom_admin_requires_full_name_when_creating_user(self):
        self.client.force_login(self.admin)
        response = self.client.post(
            reverse("admin:user_add"),
            {
                "username": "missing_name",
                "full_name": "",
                "email": "missing-name@example.com",
                "is_staff": "",
                "is_superuser": "",
                "password1": "CreatedPass123!",
                "password2": "CreatedPass123!",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertFormError(response.context["form"], "full_name", "Bidang ini tidak boleh kosong.")
        self.assertFalse(User.objects.filter(username="missing_name").exists())

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
            input_type=PredictionHistory.InputType.FILE,
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
                "input_type": PredictionHistory.InputType.FILE,
                "text_input": "",
                "source_name": "edited.csv",
                "text_column": "review",
                "sample_count": 2,
                "columns": '["review"]',
                "rows": '[{"review": "bagus", "knn_label": "Positive", "svm_label": "Positive"}]',
                "output_filename": "edited.csv",
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
                "window_days": 1,
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
            input_type=PredictionHistory.InputType.SINGLE,
            text_input="contoh kalimat",
            sample_count=1,
            rows=[{"text": "contoh kalimat", "knn_label": "Positive", "svm_label": "Positive"}],
        )

        self.client.force_login(self.user)
        self.assertContains(
            self.client.get(reverse("home")),
            "<title>Beranda | Sistem Analisis Sentimen</title>",
            html=True,
        )
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
        self.assertContains(
            self.client.get(reverse("twitter_fetch")),
            "<title>Scraping Web X | Sistem Analisis Sentimen</title>",
            html=True,
        )
        self.assertContains(
            self.client.get(reverse("history_detail", args=[scrape_history.id])),
            "<title>Detail Riwayat Scraping | Sistem Analisis Sentimen</title>",
            html=True,
        )
        self.assertContains(
            self.client.get(reverse("prediction_history_detail", args=[prediction_history.id])),
            "<title>Detail Riwayat Prediksi | Sistem Analisis Sentimen</title>",
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
            input_type=PredictionHistory.InputType.SINGLE,
            sample_count=1,
            rows=[],
        )
        PredictionHistory.objects.create(
            user=self.user,
            input_type=PredictionHistory.InputType.FILE,
            sample_count=4,
            rows=[],
        )
        PredictionHistory.objects.create(
            user=self.other_user,
            input_type=PredictionHistory.InputType.FILE,
            sample_count=88,
            rows=[],
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("home"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Ringkasan Hasil")
        self.assertContains(response, "Scraping")
        self.assertContains(response, "Prediksi")
        self.assertContains(response, "Total Scraping")
        self.assertContains(response, "Total Tweet Scraping")
        self.assertContains(response, "Total Prediksi")
        self.assertContains(response, "Total Data Prediksi")
        self.assertEqual(response.context["total_scraping_count"], 2)
        self.assertEqual(response.context["total_scraping_results"], 5)
        self.assertEqual(response.context["total_prediction_count"], 2)
        self.assertEqual(response.context["total_prediction_results"], 5)
        self.assertNotContains(response, "Website Ini Untuk Apa?")

    def test_scraping_post_creates_history_for_logged_user(self):
        self.client.force_login(self.user)
        mocked_tweets = [
            {
                "id": "101",
                "text": "mobil listrik makin bagus",
                "CreatedAt": "2026-01-01T12:34:56+00:00",
                "url": "https://x.com/test/status/101",
            }
        ]
        mocked_predictions = [
            {
                "text": "mobil listrik makin bagus",
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

        with patch("sentiment_app.views.fetch_tweets", return_value=mocked_tweets) as mocked_fetch, patch(
            "sentiment_app.views.predict_batch_in_chunks",
            return_value=mocked_predictions,
        ):
            response = self.client.post(
                reverse("twitter_fetch"),
                {
                    "api_key": "dummy_api_key",
                    "query": "mobil listrik",
                    "language": "in",
                    "start_date": "01/01/2026",
                    "end_date": "02/01/2026",
                },
            )
        self.assertEqual(mocked_fetch.call_args.kwargs.get("window_days"), 1)
        self.assertEqual(mocked_fetch.call_args.kwargs.get("max_total_tweets"), 4000)

        self.assertEqual(response.status_code, 302)
        self.assertEqual(ScrapeHistory.objects.count(), 1)
        history = ScrapeHistory.objects.get()
        self.assertEqual(history.user, self.user)
        self.assertEqual(history.tweet_count, 1)
        self.assertEqual(history.rows[0]["id"], "101")
        self.assertIn("show=1", response.url)

    def test_history_list_only_shows_owner_data(self):
        ScrapeHistory.objects.create(
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
        PredictionHistory.objects.create(
            user=self.user,
            input_type=PredictionHistory.InputType.SINGLE,
            text_input="prediksi user",
            sample_count=1,
            rows=[{"text": "prediksi user", "knn_label": "Positive", "svm_label": "Positive"}],
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("history_list"))
        self.assertContains(response, "query_user")
        self.assertContains(response, "Riwayat Prediksi")
        self.assertContains(response, "Status")
        self.assertContains(response, "Selesai", count=2)
        self.assertNotContains(response, "query_other")

        forbidden_detail = self.client.get(reverse("history_detail", args=[other_history.id]))
        self.assertEqual(forbidden_detail.status_code, 404)

    def test_scrape_history_detail_has_detail_header_and_back_button(self):
        history = ScrapeHistory.objects.create(
            user=self.user,
            query="mobil listrik",
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
        self.assertContains(response, "Detail Riwayat Scraping")
        self.assertContains(response, "Kembali ke Riwayat")
        self.assertContains(response, "Status Riwayat Scraping")
        self.assertNotContains(response, "Mulai Scraping")

    def test_predict_single_creates_prediction_history(self):
        self.client.force_login(self.user)
        mocked_result = {
            "text": "mobil listrik bagus",
            "knn_label": "Positive",
            "knn_positive_score": 0.93,
            "knn_negative_score": 0.07,
            "svm_label": "Positive",
            "svm_positive_score": 0.89,
            "svm_negative_score": 0.11,
            "combined_label": "Positive",
            "combined_positive_score": 0.91,
            "combined_negative_score": 0.09,
        }

        with patch("sentiment_app.views.predict_single", return_value=mocked_result):
            response = self.client.post(
                reverse("predict"),
                {
                    "input_mode": "single",
                    "text_input": "mobil listrik bagus",
                },
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(PredictionHistory.objects.count(), 1)
        history = PredictionHistory.objects.get()
        self.assertEqual(response.url, reverse("prediction_history_detail", args=[history.id]))
        self.assertEqual(history.user, self.user)
        self.assertEqual(history.input_type, PredictionHistory.InputType.SINGLE)
        self.assertEqual(history.sample_count, 1)
        self.assertEqual(history.rows[0]["knn_label"], "Positive")
        self.assertAlmostEqual(history.rows[0]["knn_positive_score"], 0.93, places=4)
        self.assertAlmostEqual(history.rows[0]["knn_negative_score"], 0.07, places=4)
        self.assertEqual(history.rows[0]["combined_label"], "Positive")
        self.assertAlmostEqual(history.rows[0]["combined_positive_score"], 0.91, places=4)
        self.assertAlmostEqual(history.rows[0]["combined_negative_score"], 0.09, places=4)

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
            "sentiment_app.views.parse_uploaded_file",
            return_value=(mocked_texts, "text", mocked_source_rows, mocked_source_columns),
        ), patch(
            "sentiment_app.views.predict_batch",
            return_value=mocked_predictions,
        ), patch(
            "sentiment_app.views.generate_classification_csv",
            return_value="uploaded_dummy.csv",
        ):
            response = self.client.post(
                reverse("predict"),
                {
                    "input_mode": "file",
                    "text_input": "",
                    "text_column": "",
                    "upload_file": uploaded,
                },
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(PredictionHistory.objects.count(), 1)
        history = PredictionHistory.objects.get()
        self.assertEqual(response.url, reverse("prediction_history_detail", args=[history.id]))
        self.assertEqual(history.user, self.user)
        self.assertEqual(history.input_type, PredictionHistory.InputType.FILE)
        self.assertEqual(history.source_name, "uji.csv")
        self.assertEqual(history.sample_count, 2)
        self.assertEqual(history.output_filename, "uploaded_dummy.csv")
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
            input_type=PredictionHistory.InputType.FILE,
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
        response = self.client.get(reverse("prediction_history_detail", args=[history.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Dashboard Hasil Prediksi")
        self.assertContains(response, "Sentimen Positif: 1")
        self.assertContains(response, "Sentimen Netral: 0")
        self.assertContains(response, "Sentimen Negatif: 1")
        self.assertContains(response, "prediction-dashboard-data")
        self.assertEqual(response.context["dashboard"]["charts"]["trend_title"], "Jumlah Data per Harian")

    def test_prediction_file_history_detail_hides_id_column_in_preview(self):
        history = PredictionHistory.objects.create(
            user=self.user,
            input_type=PredictionHistory.InputType.FILE,
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
            output_filename="uji_hasil.csv",
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
        self.assertContains(response, "Unduh CSV Lengkap")

    def test_prediction_history_detail_is_user_scoped(self):
        own_history = PredictionHistory.objects.create(
            user=self.user,
            input_type=PredictionHistory.InputType.SINGLE,
            text_input="tes",
            sample_count=1,
            rows=[{"text": "tes", "knn_label": "Positive", "svm_label": "Positive"}],
        )
        other_history = PredictionHistory.objects.create(
            user=self.other_user,
            input_type=PredictionHistory.InputType.SINGLE,
            text_input="other",
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
            input_type=PredictionHistory.InputType.FILE,
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

    def test_prediction_history_detail_upgrades_legacy_svm_scores(self):
        history = PredictionHistory.objects.create(
            user=self.user,
            input_type=PredictionHistory.InputType.FILE,
            source_name="legacy.csv",
            text_column="review",
            sample_count=1,
            score_schema_version=1,
            rows=[
                {
                    "review": "mobil listrik bagus",
                    "knn_label": "Positive",
                    "knn_score": 0.91,
                    "svm_label": "Positive",
                    "svm_score": 0.7,
                }
            ],
            output_filename="legacy.csv",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_dir = Path(tmp_dir) / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            (outputs_dir / "legacy.csv").write_text("old", encoding="utf-8")

            self.client.force_login(self.user)
            with self.settings(MEDIA_ROOT=tmp_dir):
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
                refreshed_csv = (outputs_dir / "legacy.csv").read_text(encoding="utf-8")

        history.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(history.score_schema_version, 7)
        self.assertEqual(history.rows[0]["svm_label"], "Positive")
        self.assertAlmostEqual(history.rows[0]["svm_positive_score"], 0.7311, places=4)
        self.assertAlmostEqual(history.rows[0]["svm_negative_score"], 0.2689, places=4)
        self.assertNotIn("svm_score", history.rows[0])
        self.assertEqual(history.rows[0]["combined_label"], "Positive")
        self.assertAlmostEqual(history.rows[0]["combined_positive_score"], 0.8205, places=4)
        self.assertAlmostEqual(history.rows[0]["combined_negative_score"], 0.1795, places=4)
        self.assertIn("0.731100", refreshed_csv)
        self.assertContains(response, "0,7311")

    def test_prediction_history_detail_reupgrades_version_two_history(self):
        history = PredictionHistory.objects.create(
            user=self.user,
            input_type=PredictionHistory.InputType.FILE,
            source_name="version2.csv",
            text_column="review",
            sample_count=1,
            score_schema_version=2,
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
        self.assertEqual(history.score_schema_version, 7)
        self.assertEqual(history.rows[0]["svm_label"], "Neutral")
        self.assertAlmostEqual(history.rows[0]["svm_positive_score"], 0.5, places=4)
        self.assertAlmostEqual(history.rows[0]["svm_negative_score"], 0.5, places=4)
        self.assertEqual(history.rows[0]["combined_label"], "Neutral")
        self.assertAlmostEqual(history.rows[0]["combined_positive_score"], 0.5, places=4)
        self.assertAlmostEqual(history.rows[0]["combined_negative_score"], 0.5, places=4)

    def test_prediction_history_detail_reupgrades_version_six_history_for_new_neutral_threshold(self):
        history = PredictionHistory.objects.create(
            user=self.user,
            input_type=PredictionHistory.InputType.FILE,
            source_name="version6.csv",
            text_column="review",
            sample_count=1,
            score_schema_version=6,
            rows=[
                {
                    "review": "teks agak netral",
                    "knn_label": "Positive",
                    "knn_positive_score": 0.58,
                    "knn_negative_score": 0.42,
                    "svm_label": "Positive",
                    "svm_positive_score": 0.58,
                    "svm_negative_score": 0.42,
                    "combined_label": "Positive",
                    "combined_positive_score": 0.58,
                    "combined_negative_score": 0.42,
                }
            ],
        )

        self.client.force_login(self.user)
        with patch(
            "sentiment_app.views.predict_batch",
            return_value=[
                {
                    "text": "teks agak netral",
                    "knn_label": "Neutral",
                    "knn_positive_score": 0.58,
                    "knn_negative_score": 0.42,
                    "svm_label": "Neutral",
                    "svm_positive_score": 0.58,
                    "svm_negative_score": 0.42,
                    "combined_label": "Neutral",
                    "combined_positive_score": 0.58,
                    "combined_negative_score": 0.42,
                }
            ],
        ):
            response = self.client.get(reverse("prediction_history_detail", args=[history.id]))

        history.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(history.score_schema_version, 7)
        self.assertEqual(history.rows[0]["knn_label"], "Neutral")
        self.assertEqual(history.rows[0]["svm_label"], "Neutral")
        self.assertEqual(history.rows[0]["combined_label"], "Neutral")

    def test_download_output_upgrades_legacy_prediction_history_csv(self):
        history = PredictionHistory.objects.create(
            user=self.user,
            input_type=PredictionHistory.InputType.FILE,
            source_name="legacy.csv",
            text_column="review",
            sample_count=1,
            score_schema_version=1,
            rows=[
                {
                    "review": "servis buruk",
                    "knn_label": "Negative",
                    "knn_score": 0.12,
                    "svm_label": "Negative",
                    "svm_score": -0.4,
                }
            ],
            output_filename="legacy.csv",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_dir = Path(tmp_dir) / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            (outputs_dir / "legacy.csv").write_text("outdated", encoding="utf-8")

            self.client.force_login(self.user)
            with self.settings(MEDIA_ROOT=tmp_dir):
                with patch(
                    "sentiment_app.views.predict_batch",
                    return_value=[
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
                        }
                    ],
                ):
                    response = self.client.get(reverse("download_output", args=["legacy.csv"]))
                    downloaded_csv = b"".join(response.streaming_content).decode("utf-8")

        history.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(history.score_schema_version, 7)
        self.assertAlmostEqual(history.rows[0]["svm_positive_score"], 0.4013, places=4)
        self.assertAlmostEqual(history.rows[0]["svm_negative_score"], 0.5987, places=4)
        self.assertEqual(history.rows[0]["combined_label"], "Negative")
        self.assertAlmostEqual(history.rows[0]["combined_positive_score"], 0.2607, places=4)
        self.assertAlmostEqual(history.rows[0]["combined_negative_score"], 0.7393, places=4)
        self.assertIn("0.401300", downloaded_csv)
        self.assertIn("0.260700", downloaded_csv)

    def test_scrape_history_detail_upgrades_legacy_svm_scores_in_chunks(self):
        history = ScrapeHistory.objects.create(
            user=self.user,
            query="dataset scraping",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-02",
            tweet_count=2,
            score_schema_version=1,
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
        self.assertEqual(history.score_schema_version, 7)
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
        mocked_window_tweets = [
            {
                "id": "201",
                "text": "mobil listrik hemat energi",
                "CreatedAt": "2026-01-15T12:00:00+00:00",
                "url": "https://x.com/test/status/201",
            }
        ]
        mocked_predictions = [
            {
                "text": "mobil listrik hemat energi",
                "knn_label": "Positive",
                "knn_positive_score": 0.9,
                "knn_negative_score": 0.1,
                "svm_label": "Positive",
                "svm_positive_score": 0.85,
                "svm_negative_score": 0.15,
                "combined_label": "Positive",
                "combined_positive_score": 0.875,
                "combined_negative_score": 0.125,
            }
        ]

        def _fake_fetch_tweets(*args, **kwargs):
            callback = kwargs.get("on_window")
            if callback:
                callback(mocked_window_tweets)
            return []

        with patch("sentiment_app.views.fetch_tweets", side_effect=_fake_fetch_tweets), patch(
            "sentiment_app.views.predict_batch_in_chunks",
            return_value=mocked_predictions,
        ):
            response = self.client.post(
                reverse("twitter_fetch"),
                {
                    "api_key": "dummy_api_key",
                    "query": "mobil listrik",
                    "language": "in",
                    "start_date": "01/01/2026",
                    "end_date": "05/03/2026",
                },
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(ScrapeHistory.objects.count(), 1)
        history = ScrapeHistory.objects.get()
        self.assertEqual(history.tweet_count, 1)
        self.assertEqual(history.rows, [])
        self.assertEqual(ScrapeTempChunk.objects.filter(history=history).count(), 1)
        self.assertIn(f"history={history.id}", response.url)

    def test_scraping_marks_history_incomplete_when_partial_timeout(self):
        self.client.force_login(self.user)
        mocked_tweets = [
            {
                "id": "111",
                "text": "uji timeout parsial",
                "CreatedAt": "2026-01-01T10:00:00+00:00",
            }
        ]
        mocked_predictions = [
            {
                "text": "uji timeout parsial",
                "knn_label": "Positive",
                "knn_positive_score": 0.9,
                "knn_negative_score": 0.1,
                "svm_label": "Positive",
                "svm_positive_score": 0.8,
                "svm_negative_score": 0.2,
                "combined_label": "Positive",
                "combined_positive_score": 0.85,
                "combined_negative_score": 0.15,
            }
        ]
        mocked_meta = {
            "rate_limited": False,
            "timed_out": True,
            "truncated": False,
            "next_start_date": "2026-01-02",
        }

        with patch("sentiment_app.views.fetch_tweets", return_value=(mocked_tweets, mocked_meta)), patch(
            "sentiment_app.views.predict_batch_in_chunks",
            return_value=mocked_predictions,
        ):
            response = self.client.post(
                reverse("twitter_fetch"),
                {
                    "api_key": "dummy_api_key",
                    "query": "mobil listrik",
                    "language": "in",
                    "start_date": "01/01/2026",
                    "end_date": "03/01/2026",
                },
            )

        self.assertEqual(response.status_code, 302)
        history = ScrapeHistory.objects.get()
        self.assertFalse(history.is_complete)
        self.assertEqual(str(history.resume_next_date), "2026-01-02")
        self.assertEqual(history.stop_reason, "timed_out")

    def test_resume_scrape_appends_rows_and_can_complete(self):
        self.client.force_login(self.user)
        history = ScrapeHistory.objects.create(
            user=self.user,
            query="mobil listrik",
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
            window_days=1,
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
        self.assertTrue(history.is_complete)
        self.assertIsNone(history.resume_next_date)
        self.assertEqual(history.tweet_count, 2)
        self.assertEqual(ScrapeTempChunk.objects.filter(history=history).count(), 1)

    def test_resume_scrape_ajax_returns_progress_payload(self):
        self.client.force_login(self.user)
        history = ScrapeHistory.objects.create(
            user=self.user,
            query="mobil listrik",
            language="in",
            start_date="2026-01-01",
            end_date="2026-01-03",
            tweet_count=0,
            rows=[],
            is_complete=False,
            resume_next_date="2026-01-02",
            stop_reason="timed_out",
            window_days=1,
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
            window_days=1,
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
            window_days=1,
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
