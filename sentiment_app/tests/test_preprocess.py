from django.test import SimpleTestCase
from unittest.mock import patch

from sentiment_app.services import model_service
from sentiment_app.services.model_service import preprocess_text


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
        self.assertTrue("perjalanan" in cleaned or "jalan" in cleaned)
        self.assertNotIn("123", cleaned)

    def test_preprocess_can_skip_stemming(self):
        class FakeStemmer:
            def stem(self, word):
                return f"stem_{word}"

        with patch("sentiment_app.services.model_service._get_stemmer", return_value=FakeStemmer()):
            with_stemming = preprocess_text("rumah besar", apply_stemming=True)
            without_stemming = preprocess_text("rumah besar", apply_stemming=False)

        self.assertIn("stem_rumah", with_stemming)
        self.assertEqual(without_stemming, "rumah besar")

    def test_preprocess_uses_singkatan_tsv_from_models_dir(self):
        previous_cache = model_service._SLANG_MAP_CACHE
        self.addCleanup(setattr, model_service, "_SLANG_MAP_CACHE", previous_cache)

        with self.settings(SENTIMENT_MODELS_DIR=self._tmp_models_dir()):
            model_service._SLANG_MAP_CACHE = None
            cleaned = preprocess_text("abcx bagus", apply_stemming=False)

        self.assertIn("istimewaunik", cleaned)

    def _tmp_models_dir(self):
        import tempfile
        from pathlib import Path

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        models_dir = Path(temp_dir.name)
        (models_dir / "singkatan.tsv").write_text("abcx\tistimewaunik\n", encoding="utf-8")
        return models_dir
