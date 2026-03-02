from unittest.mock import patch

from django.test import SimpleTestCase

from sentiment_app.services.twitter_client import TwitterAPIError, _fetch_window_tweets, fetch_tweets


class TwitterClientServiceTests(SimpleTestCase):
    @patch("sentiment_app.services.twitter_client.time.sleep", return_value=None)
    def test_fetch_tweets_keeps_rows_with_unparseable_created_at(self, _mock_sleep):
        window_1 = [{"id": "1", "text": "tweet hari 1"}]
        window_2 = [{"id": "2", "text": "tweet hari 2"}]
        window_3 = [{"id": "3", "text": "tweet hari 3"}]

        with patch(
            "sentiment_app.services.twitter_client._fetch_window_tweets",
            side_effect=[window_1, window_2, window_3],
        ):
            tweets = fetch_tweets(
                api_key="dummy",
                query="kendaraan listrik",
                start_date="2026-02-01",
                end_date="2026-02-03",
                window_days=1,
                max_tweets_per_window=10,
                max_total_tweets=50,
            )

        self.assertEqual(len(tweets), 3)
        self.assertEqual(tweets[0]["_week_start"], "2026-02-01")
        self.assertEqual(tweets[1]["_week_start"], "2026-02-02")
        self.assertEqual(tweets[2]["_week_start"], "2026-02-03")

    @patch("sentiment_app.services.twitter_client.time.sleep", return_value=None)
    def test_fetch_tweets_filters_by_global_range(self, _mock_sleep):
        # Window 1 returns a tweet from day-2 (still inside global range) -> should be kept.
        # Window 2 returns tweet outside global range -> should be dropped.
        window_1 = [{"id": "A", "text": "inside global", "CreatedAt": "2026-02-02T10:00:00+00:00"}]
        window_2 = [{"id": "B", "text": "outside global", "CreatedAt": "2026-02-05T10:00:00+00:00"}]
        window_3 = []

        with patch(
            "sentiment_app.services.twitter_client._fetch_window_tweets",
            side_effect=[window_1, window_2, window_3],
        ):
            tweets = fetch_tweets(
                api_key="dummy",
                query="kendaraan listrik",
                start_date="2026-02-01",
                end_date="2026-02-03",
                window_days=1,
                max_tweets_per_window=10,
                max_total_tweets=50,
            )

        self.assertEqual(len(tweets), 1)
        self.assertEqual(tweets[0]["id"], "A")

    @patch("sentiment_app.services.twitter_client.time.sleep", return_value=None)
    def test_fetch_window_tweets_raises_on_rate_limit_after_retries(self, _mock_sleep):
        class DummyResponse:
            status_code = 429

            @staticmethod
            def json():
                return {"message": "rate limit"}

        with patch("sentiment_app.services.twitter_client.requests.get", return_value=DummyResponse()):
            with self.assertRaises(TwitterAPIError):
                _fetch_window_tweets(
                    api_key="dummy",
                    query="kendaraan listrik",
                    max_tweets_per_window=5,
                )
