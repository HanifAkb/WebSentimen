from django.test import SimpleTestCase

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

    def test_rejects_when_start_date_after_end_date(self):
        form = TwitterFetchForm(
            data={
                "api_key": "dummy",
                "query": "mobil listrik",
                "language": "in",
                "start_date": "10/01/2026",
                "end_date": "09/01/2026",
            }
        )
        self.assertFalse(form.is_valid())
        self.assertIn("Tanggal mulai tidak boleh lebih besar", str(form.non_field_errors()))

    def test_accepts_iso_date_format_from_native_date_input(self):
        form = TwitterFetchForm(
            data={
                "api_key": "dummy",
                "query": "mobil listrik",
                "language": "in",
                "start_date": "2026-01-01",
                "end_date": "2026-01-07",
            }
        )
        self.assertTrue(form.is_valid(), form.errors)

    def test_accepts_empty_language_for_all_languages(self):
        form = TwitterFetchForm(
            data={
                "api_key": "dummy",
                "query": "mobil listrik",
                "language": "",
                "start_date": "01/01/2026",
                "end_date": "07/01/2026",
            }
        )

        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data["language"], "")

    def test_rejects_language_outside_dropdown_choices(self):
        form = TwitterFetchForm(
            data={
                "api_key": "dummy",
                "query": "mobil listrik",
                "language": "jp",
                "start_date": "01/01/2026",
                "end_date": "07/01/2026",
            }
        )

        self.assertFalse(form.is_valid())
        self.assertIn("language", form.errors)
