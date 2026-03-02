from django.test import SimpleTestCase, override_settings

from sentiment_app.forms import TwitterFetchForm


class TwitterFetchFormTests(SimpleTestCase):
    def test_accepts_range_within_limit(self):
        form = TwitterFetchForm(
            data={
                "api_key": "dummy",
                "query": "mobil listrik",
                "language": "in",
                "start_date": "01/01/2026",
                "end_date": "07/01/2026",
            }
        )
        self.assertTrue(form.is_valid(), form.errors)

    @override_settings(SENTIMENT_TWITTER_MAX_RANGE_DAYS=7)
    def test_rejects_range_over_limit(self):
        form = TwitterFetchForm(
            data={
                "api_key": "dummy",
                "query": "mobil listrik",
                "language": "in",
                "start_date": "01/01/2026",
                "end_date": "09/01/2026",
            }
        )
        self.assertFalse(form.is_valid())
        self.assertIn("Maksimal 7 hari", str(form.non_field_errors()))

    def test_rejects_iso_date_format(self):
        form = TwitterFetchForm(
            data={
                "api_key": "dummy",
                "query": "mobil listrik",
                "language": "in",
                "start_date": "2026-01-01",
                "end_date": "2026-01-07",
            }
        )
        self.assertFalse(form.is_valid())
        self.assertIn("Format tanggal harus dd/mm/yyyy", str(form.errors))
