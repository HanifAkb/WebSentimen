from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

import sentiment_app.views as view_module
from sentiment_app.views import _build_prediction_dashboard, _build_scraping_dashboard, _build_wordcloud_image


class _WordCloudImageStub:
    def save(self, buffer, format="PNG"):
        buffer.write(b"png")


class _WordCloudStub:
    last_frequencies = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate_from_frequencies(self, frequencies):
        type(self).last_frequencies = dict(frequencies)
        return self

    def to_image(self):
        return _WordCloudImageStub()


class ScrapingDashboardTests(SimpleTestCase):
    def test_weekly_trend_uses_start_date_as_bucket_anchor(self):
        rows = [
            {"CreatedAt": "2026-02-01T10:00:00+00:00", "text": "a", "knn_label": "Positive", "svm_label": "Positive", "combined_label": "Positive"},
            {"CreatedAt": "2026-02-02T10:00:00+00:00", "text": "b", "knn_label": "Negative", "svm_label": "Negative", "combined_label": "Negative"},
            {"CreatedAt": "2026-02-10T10:00:00+00:00", "text": "c", "knn_label": "Positive", "svm_label": "Positive", "combined_label": "Positive"},
            {"CreatedAt": "2026-02-20T10:00:00+00:00", "text": "d", "knn_label": "Negative", "svm_label": "Negative", "combined_label": "Negative"},
        ]

        dashboard = _build_scraping_dashboard(rows, date(2026, 2, 1), date(2026, 2, 20))
        charts = dashboard["charts"]

        self.assertEqual(charts["trend_values"], [2, 1, 1])
        self.assertEqual(len(charts["trend_labels"]), 3)
        self.assertTrue(str(charts["trend_labels"][0]).startswith("01 "))

    def test_dashboard_accepts_unix_timestamp_created_at(self):
        rows = [
            {"CreatedAt": "1706745600", "text": "a", "knn_label": "Positive", "svm_label": "Positive", "combined_label": "Positive"},
        ]
        dashboard = _build_scraping_dashboard(rows, date(2024, 2, 1), date(2024, 2, 1))
        charts = dashboard["charts"]
        self.assertEqual(charts["trend_values"], [1])
        self.assertEqual(dashboard["period_label"], "01 Februari 2024 - 01 Februari 2024")

    @override_settings(SENTIMENT_WORDCLOUD_MAX_ROWS=1)
    def test_prediction_dashboard_uses_history_text_column_and_data_title(self):
        history = SimpleNamespace(
            text_column="review",
            created_at=datetime(2026, 1, 10, 9, 0, tzinfo=timezone.utc),
        )
        rows = [
            {
                "review": "mobil listrik sangat bagus",
                "tanggal": "2026-01-01",
                "knn_label": "Positive",
                "svm_label": "Positive",
                "combined_label": "Positive",
            },
            {
                "review": "servis kendaraan buruk",
                "tanggal": "2026-01-02",
                "knn_label": "Negative",
                "svm_label": "Negative",
                "combined_label": "Negative",
            },
        ]

        dashboard = _build_prediction_dashboard(history, rows, ["review", "tanggal"])

        self.assertEqual(dashboard["charts"]["knn_pie"], [1, 1, 0])
        self.assertEqual(dashboard["charts"]["svm_pie"], [1, 1, 0])
        self.assertEqual(dashboard["charts"]["combined_pie"], [1, 1, 0])
        self.assertIn("combined_positive_image", dashboard["wordclouds"])
        self.assertIn("combined_negative_image", dashboard["wordclouds"])
        self.assertNotIn("knn_positive_image", dashboard["wordclouds"])
        self.assertNotIn("svm_positive_image", dashboard["wordclouds"])
        self.assertEqual(dashboard["charts"]["trend_title"], "Jumlah Data per Harian")
        self.assertEqual(dashboard["charts"]["trend_values"], [1, 1])

    def test_dashboard_exposes_top_unigram_stats_for_wordcloud(self):
        rows = [
            {"CreatedAt": "2026-02-01T10:00:00+00:00", "text": "mobil listrik mobil baterai hemat cepat", "knn_label": "Positive", "svm_label": "Positive", "combined_label": "Positive"},
            {"CreatedAt": "2026-02-02T10:00:00+00:00", "text": "servis buruk servis lambat mahal rusak", "knn_label": "Negative", "svm_label": "Negative", "combined_label": "Negative"},
        ]

        with patch("sentiment_app.views.WordCloud", _WordCloudStub):
            dashboard = _build_scraping_dashboard(rows, date(2026, 2, 1), date(2026, 2, 2))

        self.assertEqual(
            dashboard["wordcloud_top_unigrams"]["combined_positive"],
            [
                {"word": "mobil", "count": 2},
                {"word": "listrik", "count": 1},
                {"word": "baterai", "count": 1},
                {"word": "hemat", "count": 1},
                {"word": "cepat", "count": 1},
            ],
        )
        self.assertEqual(
            dashboard["wordcloud_top_unigrams"]["combined_negative"],
            [
                {"word": "servis", "count": 2},
                {"word": "buruk", "count": 1},
                {"word": "lambat", "count": 1},
                {"word": "mahal", "count": 1},
                {"word": "rusak", "count": 1},
            ],
        )

    def test_dashboard_excludes_query_terms_from_wordcloud_and_unigrams(self):
        rows = [
            {"CreatedAt": "2026-02-01T10:00:00+00:00", "text": "kendaraan listrik hemat kendaraan listrik cepat baterai", "knn_label": "Positive", "svm_label": "Positive", "combined_label": "Positive"},
            {"CreatedAt": "2026-02-02T10:00:00+00:00", "text": "kendaraan listrik mahal servis buruk", "knn_label": "Negative", "svm_label": "Negative", "combined_label": "Negative"},
        ]

        with patch("sentiment_app.views.WordCloud", _WordCloudStub), patch(
            "sentiment_app.views._load_wordcloud_stopwords",
            return_value=set(),
        ):
            dashboard = _build_scraping_dashboard(
                rows,
                date(2026, 2, 1),
                date(2026, 2, 2),
                query='"kendaraan listrik"',
            )

        self.assertEqual(
            dashboard["wordcloud_top_unigrams"]["combined_positive"],
            [
                {"word": "hemat", "count": 1},
                {"word": "cepat", "count": 1},
                {"word": "baterai", "count": 1},
            ],
        )
        self.assertEqual(
            dashboard["wordcloud_top_unigrams"]["combined_negative"],
            [
                {"word": "mahal", "count": 1},
                {"word": "servis", "count": 1},
                {"word": "buruk", "count": 1},
            ],
        )
        self.assertNotIn("kendara", _WordCloudStub.last_frequencies)
        self.assertNotIn("listrik", _WordCloudStub.last_frequencies)

    def test_wordcloud_uses_unigram_frequencies(self):
        with patch("sentiment_app.views.WordCloud", _WordCloudStub):
            result = _build_wordcloud_image(["mobil listrik bagus"], colormap="Greens")

        self.assertIsNotNone(result)
        self.assertEqual(_WordCloudStub.last_frequencies["mobil"], 1)
        self.assertEqual(_WordCloudStub.last_frequencies["listrik"], 1)
        self.assertEqual(_WordCloudStub.last_frequencies["bagus"], 1)
        self.assertNotIn("mobil listrik", _WordCloudStub.last_frequencies)
        self.assertNotIn("listrik bagus", _WordCloudStub.last_frequencies)

    def test_wordcloud_uses_preprocessed_text_with_stemming_pipeline(self):
        with patch("sentiment_app.views.WordCloud", _WordCloudStub), patch(
            "sentiment_app.views.preprocess_text",
            return_value="kendara listrik baik",
        ) as mocked_preprocess, patch(
            "sentiment_app.views._load_wordcloud_stopwords",
            return_value=set(),
        ):
            result = _build_wordcloud_image(["Kendaraan Listrik terbaik!!!"], colormap="Greens")

        self.assertIsNotNone(result)
        mocked_preprocess.assert_called_once_with("Kendaraan Listrik terbaik!!!", apply_stemming=True)
        self.assertEqual(_WordCloudStub.last_frequencies["kendara"], 1)
        self.assertEqual(_WordCloudStub.last_frequencies["listrik"], 1)
        self.assertEqual(_WordCloudStub.last_frequencies["baik"], 1)
        self.assertNotIn("kendaraan", _WordCloudStub.last_frequencies)
        self.assertNotIn("terbaik", _WordCloudStub.last_frequencies)

    def test_wordcloud_counter_still_uses_wordcloud_stopword_file(self):
        with patch("sentiment_app.views.preprocess_text", return_value="mobil dan listrik hemat"), patch(
            "sentiment_app.views._load_wordcloud_stopwords",
            return_value={"dan", "hemat"},
        ):
            frequencies = view_module._build_wordcloud_unigram_counter(["Mobil dan listrik hemat"])

        self.assertEqual(frequencies, {"mobil": 1, "listrik": 1})
        self.assertNotIn("dan", frequencies)
        self.assertNotIn("hemat", frequencies)

    def test_wordcloud_stopwords_reload_when_file_changes(self):
        with TemporaryDirectory() as temp_dir:
            stopwords_path = Path(temp_dir) / "stopwords-id(wordcloud).txt"
            stopwords_path.write_text("anda\n", encoding="utf-8")

            with patch.object(view_module, "WORDCLOUD_STOPWORDS_PATHS", [stopwords_path]):
                view_module._WORDCLOUD_STOPWORDS_CACHE = None
                view_module._WORDCLOUD_STOPWORDS_CACHE_SIGNATURE = None

                first_load = view_module._load_wordcloud_stopwords()
                self.assertEqual(first_load, {"anda"})

                stopwords_path.write_text("lagi\n", encoding="utf-8")

                second_load = view_module._load_wordcloud_stopwords()
                self.assertEqual(second_load, {"lagi"})
