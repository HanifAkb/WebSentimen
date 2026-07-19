from datetime import date
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
    def test_fetch_window_tweets_skips_duplicate_tweet_ids_and_tolerates_duplicate_page(self, _mock_sleep):
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
                    "tweets": [{"id": "1", "text": "pertama ulang"}, {"id": "2", "text": "kedua ulang"}],
                    "has_next_page": True,
                    "next_cursor": "cursor-2",
                }
            ),
            DummyResponse(
                {
                    "tweets": [{"id": "3", "text": "ketiga"}],
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
    def test_fetch_window_tweets_skips_existing_ids_and_out_of_range_before_consuming_quota(self, _mock_sleep):
        class DummyResponse:
            status_code = 200

            def __init__(self, payload):
                self.payload = payload

            def json(self):
                return self.payload

        responses = [
            DummyResponse(
                {
                    "tweets": [
                        {"id": "1", "text": "duplikat lama", "CreatedAt": "2026-02-01T10:00:00+00:00"},
                        {"id": "9", "text": "di luar hari", "CreatedAt": "2026-02-03T10:00:00+00:00"},
                    ],
                    "has_next_page": True,
                    "next_cursor": "cursor-1",
                }
            ),
            DummyResponse(
                {
                    "tweets": [
                        {"id": "2", "text": "tweet hari benar", "CreatedAt": "2026-02-02T10:00:00+00:00"},
                    ],
                    "has_next_page": False,
                }
            ),
        ]

        with patch("sentiment_app.services.scraping_service.requests.get", side_effect=responses) as mocked_get:
            tweets = _fetch_window_tweets(
                api_key="dummy",
                query="kendaraan listrik",
                max_tweets_per_window=1,
                existing_tweet_keys={"id:1"},
                min_created_date=date(2026, 2, 2),
                max_created_date=date(2026, 2, 2),
            )

        self.assertEqual([tweet["id"] for tweet in tweets], ["2"])
        self.assertEqual(mocked_get.call_count, 2)

    @patch("sentiment_app.services.scraping_service.time.sleep", return_value=None)
    def test_fetch_tweets_does_not_share_raw_duplicate_state_across_windows(self, _mock_sleep):
        captured_seen_keys = []

        def _fake_fetch_window_tweets(*args, **kwargs):
            captured_seen_keys.append(kwargs.get("seen_tweet_keys"))
            query = kwargs.get("query", "")
            if "since:2026-02-01 until:2026-02-02" in query:
                return [{"id": "1", "text": "tweet hari 1", "CreatedAt": "2026-02-01T10:00:00+00:00"}]
            return [{"id": "2", "text": "tweet hari 2", "CreatedAt": "2026-02-02T10:00:00+00:00"}]

        with patch(
            "sentiment_app.services.scraping_service._fetch_window_tweets",
            side_effect=_fake_fetch_window_tweets,
        ):
            tweets = fetch_tweets(
                api_key="dummy",
                query="kendaraan listrik",
                start_date="2026-02-01",
                end_date="2026-02-02",
                window_days=1,
                max_tweets_per_window=10,
                max_total_tweets=50,
            )

        self.assertEqual([tweet["id"] for tweet in tweets], ["1", "2"])
        self.assertEqual(captured_seen_keys, [None, None])

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

    @patch("sentiment_app.services.scraping_service.time.sleep", return_value=None)
    def test_fetch_tweets_calls_checkpoint_after_each_window(self, _mock_sleep):
        window_1 = [{"id": "1", "text": "tweet hari 1", "CreatedAt": "2026-02-01T10:00:00+00:00"}]
        window_2 = []
        checkpoints = []

        with patch(
            "sentiment_app.services.scraping_service._fetch_window_tweets",
            side_effect=[window_1, window_2],
        ):
            tweets = fetch_tweets(
                api_key="dummy",
                query="kendaraan listrik",
                start_date="2026-02-01",
                end_date="2026-02-02",
                window_days=1,
                max_tweets_per_window=10,
                max_total_tweets=50,
                on_window_checkpoint=checkpoints.append,
            )

        self.assertEqual([tweet["id"] for tweet in tweets], ["1"])
        self.assertEqual(
            checkpoints,
            [
                {
                    "window_start": "2026-02-01",
                    "window_end": "2026-02-02",
                    "next_start_date": "2026-02-02",
                    "window_kept_count": 1,
                    "kept_total": 1,
                },
                {
                    "window_start": "2026-02-02",
                    "window_end": "2026-02-03",
                    "next_start_date": "2026-02-03",
                    "window_kept_count": 0,
                    "kept_total": 1,
                },
            ],
        )
