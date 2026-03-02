from django.test import SimpleTestCase
from unittest.mock import patch

from sentiment_app.services.preprocess import preprocess_text


class PreprocessTests(SimpleTestCase):
    def test_preprocess_removes_url_user_and_hash(self):
        text = "Hello @User check https://example.com #Awesome"
        cleaned = preprocess_text(text)
        self.assertEqual(cleaned, "hello check")

    def test_preprocess_normalizes_whitespace(self):
        text = "  This   is\n\nA   TEST  "
        cleaned = preprocess_text(text)
        self.assertEqual(cleaned, "this is a test")

    def test_preprocess_normalizes_slang_and_removes_numbers(self):
        text = "Gw OTW 123 ke rumah!!!"
        cleaned = preprocess_text(text)
        self.assertIn("sedang", cleaned)
        self.assertTrue("perjalanan" in cleaned or "jalan" in cleaned)
        self.assertNotIn("123", cleaned)

    def test_preprocess_can_skip_stemming(self):
        class FakeStemmer:
            def stem(self, word):
                return f"stem_{word}"

        with patch("sentiment_app.services.preprocess._get_stemmer", return_value=FakeStemmer()):
            with_stemming = preprocess_text("rumah besar", apply_stemming=True)
            without_stemming = preprocess_text("rumah besar", apply_stemming=False)

        self.assertIn("stem_rumah", with_stemming)
        self.assertEqual(without_stemming, "rumah besar")
