from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from sentiment_app.models import PredictionHistory, ScrapeHistory, ScrapeTempChunk
from sentiment_app.services.twitter_client import TwitterRateLimitError, TwitterTimeoutError


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
        self.assertContains(response, "Total Hasil Scraping")
        self.assertContains(response, "Total Hasil Prediksi")
        self.assertEqual(response.context["total_scraping_results"], 5)
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
                "knn_score": 0.91,
                "svm_label": "Positive",
                "svm_score": 0.88,
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

        self.client.force_login(self.user)
        response = self.client.get(reverse("history_list"))
        self.assertContains(response, "query_user")
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
            "knn_score": 0.93,
            "svm_label": "Positive",
            "svm_score": 0.89,
        }

        with patch("sentiment_app.views.predict_single", return_value=mocked_result):
            response = self.client.post(
                reverse("predict"),
                {
                    "input_mode": "single",
                    "text_input": "mobil listrik bagus",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(PredictionHistory.objects.count(), 1)
        history = PredictionHistory.objects.get()
        self.assertEqual(history.user, self.user)
        self.assertEqual(history.input_type, PredictionHistory.InputType.SINGLE)
        self.assertEqual(history.sample_count, 1)
        self.assertEqual(history.rows[0]["knn_label"], "Positive")

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
                "knn_score": 0.91,
                "svm_label": "Positive",
                "svm_score": 0.87,
            },
            {
                "knn_label": "Negative",
                "knn_score": 0.16,
                "svm_label": "Negative",
                "svm_score": 0.12,
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

        self.assertEqual(response.status_code, 200)
        self.assertEqual(PredictionHistory.objects.count(), 1)
        history = PredictionHistory.objects.get()
        self.assertEqual(history.user, self.user)
        self.assertEqual(history.input_type, PredictionHistory.InputType.FILE)
        self.assertEqual(history.source_name, "uji.csv")
        self.assertEqual(history.sample_count, 2)
        self.assertEqual(history.output_filename, "uploaded_dummy.csv")
        self.assertEqual(history.rows[0]["text"], "mobil listrik bagus")
        self.assertEqual(history.rows[1]["svm_label"], "Negative")

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
                    "knn_score": 0.91,
                    "svm_label": "Positive",
                    "svm_score": 0.87,
                },
                {
                    "review": "servis buruk",
                    "tanggal": "2026-01-02",
                    "knn_label": "Negative",
                    "knn_score": 0.16,
                    "svm_label": "Negative",
                    "svm_score": 0.12,
                },
            ],
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("prediction_history_detail", args=[history.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Dashboard Hasil Prediksi")
        self.assertContains(response, "KNN Positif: 1")
        self.assertContains(response, "prediction-dashboard-data")
        self.assertEqual(response.context["dashboard"]["charts"]["trend_title"], "Jumlah Data per Harian")

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
                "knn_score": 0.9,
                "svm_label": "Positive",
                "svm_score": 0.85,
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
                "knn_score": 0.9,
                "svm_label": "Positive",
                "svm_score": 0.8,
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
                "knn_score": 0.2,
                "svm_label": "Negative",
                "svm_score": 0.3,
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
                "knn_score": 0.7,
                "svm_label": "Positive",
                "svm_score": 0.6,
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
