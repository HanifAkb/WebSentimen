from datetime import date, datetime, timezone
from types import SimpleNamespace

from django.test import SimpleTestCase, override_settings

from sentiment_app.views import _build_prediction_dashboard, _build_scraping_dashboard


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
