from unittest.mock import patch

from django.test import SimpleTestCase

from sentiment_app.services.scraping_service import TwitterAPIError, _fetch_window_tweets, fetch_tweets


class ScrapingServiceTests(SimpleTestCase):
    @patch("sentiment_app.services.scraping_service.time.sleep", return_value=None)
    def test_fetch_tweets_keeps_rows_with_unparseable_created_at(self, _mock_sleep):
        window_1 = [{"id": "1", "text": "tweet hari 1"}]
        window_2 = [{"id": "2", "text": "tweet hari 2"}]
        window_3 = [{"id": "3", "text": "tweet hari 3"}]

        with patch(
            "sentiment_app.services.scraping_service._fetch_window_tweets",
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

    @patch("sentiment_app.services.scraping_service.time.sleep", return_value=None)
    def test_fetch_tweets_filters_by_global_range(self, _mock_sleep):
        # Window 1 returns a tweet from day-2 (still inside global range) -> should be kept.
        # Window 2 returns tweet outside global range -> should be dropped.
        window_1 = [{"id": "A", "text": "inside global", "CreatedAt": "2026-02-02T10:00:00+00:00"}]
        window_2 = [{"id": "B", "text": "outside global", "CreatedAt": "2026-02-05T10:00:00+00:00"}]
        window_3 = []

        with patch(
            "sentiment_app.services.scraping_service._fetch_window_tweets",
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

    @patch("sentiment_app.services.scraping_service.time.sleep", return_value=None)
    def test_fetch_window_tweets_raises_on_rate_limit_after_retries(self, _mock_sleep):
        class DummyResponse:
            status_code = 429

            @staticmethod
            def json():
                return {"message": "rate limit"}

        with patch("sentiment_app.services.scraping_service.requests.get", return_value=DummyResponse()):
            with self.assertRaises(TwitterAPIError):
                _fetch_window_tweets(
                    api_key="dummy",
                    query="kendaraan listrik",
                    max_tweets_per_window=5,
                )

    @patch("sentiment_app.services.scraping_service.time.sleep", return_value=None)
    def test_fetch_window_tweets_skips_duplicate_tweet_ids_and_stops_duplicate_page(self, _mock_sleep):
        class DummyResponse:
            status_code = 200

            def __init__(self, payload):
                self.payload = payload

            def json(self):
                return self.payload

        responses = [
            DummyResponse(
                {
                    "tweets": [{"id": "1", "text": "pertama"}, {"id": "2", "text": "kedua"}],
                    "has_next_page": True,
                    "next_cursor": "cursor-1",
                }
            ),
            DummyResponse(
                {
                    "tweets": [{"id": "2", "text": "kedua ulang"}, {"id": "3", "text": "ketiga"}],
                    "has_next_page": True,
                    "next_cursor": "cursor-2",
                }
            ),
            DummyResponse(
                {
                    "tweets": [{"id": "1", "text": "pertama ulang"}, {"id": "3", "text": "ketiga ulang"}],
                    "has_next_page": True,
                    "next_cursor": "cursor-3",
                }
            ),
            DummyResponse(
                {
                    "tweets": [{"id": "4", "text": "tidak boleh dipanggil"}],
                    "has_next_page": False,
                }
            ),
        ]

        with patch("sentiment_app.services.scraping_service.requests.get", side_effect=responses) as mocked_get:
            tweets = _fetch_window_tweets(
                api_key="dummy",
                query="kendaraan listrik",
                max_tweets_per_window=10,
            )

        self.assertEqual([tweet["id"] for tweet in tweets], ["1", "2", "3"])
        self.assertEqual(mocked_get.call_count, 3)

    @patch("sentiment_app.services.scraping_service.time.sleep", return_value=None)
    def test_fetch_tweets_returns_partial_when_timeout_after_some_data(self, _mock_sleep):
        with patch(
            "sentiment_app.services.scraping_service._fetch_window_tweets",
            side_effect=lambda *args, **kwargs: [{"id": "x", "text": "tweet"}],
        ), patch(
            "sentiment_app.services.scraping_service.time.monotonic",
            side_effect=[0.0, 0.0, 0.0, 100.0],
        ):
            tweets, meta = fetch_tweets(
                api_key="dummy",
                query="kendaraan listrik",
                start_date="2026-02-01",
                end_date="2026-02-15",
                window_days=1,
                max_tweets_per_window=10,
                max_total_tweets=100,
                max_runtime_seconds=10,
                return_meta=True,
            )

        self.assertEqual(len(tweets), 1)
        self.assertTrue(bool(meta.get("timed_out")))

    @patch("sentiment_app.services.scraping_service.time.sleep", return_value=None)
    def test_fetch_tweets_raises_when_timeout_before_any_data(self, _mock_sleep):
        with patch(
            "sentiment_app.services.scraping_service.time.monotonic",
            side_effect=[0.0, 100.0],
        ):
            with self.assertRaises(TwitterAPIError):
                fetch_tweets(
                    api_key="dummy",
                    query="kendaraan listrik",
                    start_date="2026-02-01",
                    end_date="2026-02-15",
                    window_days=1,
                    max_tweets_per_window=10,
                    max_total_tweets=100,
                    max_runtime_seconds=10,
                )
