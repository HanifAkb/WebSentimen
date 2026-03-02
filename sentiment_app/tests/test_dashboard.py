from datetime import date

from django.test import SimpleTestCase

from sentiment_app.views import _build_scraping_dashboard


class ScrapingDashboardTests(SimpleTestCase):
    def test_weekly_trend_uses_start_date_as_bucket_anchor(self):
        rows = [
            {"CreatedAt": "2026-02-01T10:00:00+00:00", "text": "a", "knn_label": "Positive", "svm_label": "Positive"},
            {"CreatedAt": "2026-02-02T10:00:00+00:00", "text": "b", "knn_label": "Negative", "svm_label": "Negative"},
            {"CreatedAt": "2026-02-10T10:00:00+00:00", "text": "c", "knn_label": "Positive", "svm_label": "Positive"},
            {"CreatedAt": "2026-02-20T10:00:00+00:00", "text": "d", "knn_label": "Negative", "svm_label": "Negative"},
        ]

        dashboard = _build_scraping_dashboard(rows, date(2026, 2, 1), date(2026, 2, 20))
        charts = dashboard["charts"]

        self.assertEqual(charts["trend_values"], [2, 1, 1])
        self.assertEqual(len(charts["trend_labels"]), 3)
        self.assertTrue(str(charts["trend_labels"][0]).startswith("01 "))
