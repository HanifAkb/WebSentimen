from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from sentiment_app.models import ScrapeHistory


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

    def test_non_admin_cannot_open_register_page(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("register_user"))
        self.assertEqual(response.status_code, 403)

    def test_admin_can_register_new_user(self):
        self.client.force_login(self.admin)
        response = self.client.post(
            reverse("register_user"),
            {
                "username": "company_user",
                "email": "company_user@example.com",
                "is_staff": "on",
                "password1": "StrongPass123!",
                "password2": "StrongPass123!",
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(User.objects.filter(username="company_user", is_staff=True).exists())

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

        with patch("sentiment_app.views.fetch_tweets", return_value=mocked_tweets), patch(
            "sentiment_app.views.predict_batch",
            return_value=mocked_predictions,
        ):
            response = self.client.post(
                reverse("twitter_fetch"),
                {
                    "api_key": "dummy_api_key",
                    "query": "mobil listrik",
                    "language": "in",
                    "start_date": "2026-01-01",
                    "end_date": "2026-01-02",
                },
            )

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
