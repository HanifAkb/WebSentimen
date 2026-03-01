import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase

from sentiment_app.services.file_service import (
    FileValidationError,
    detect_text_column,
    generate_classification_csv,
    parse_uploaded_file,
    read_csv_page,
)


class FileServiceTests(SimpleTestCase):
    def test_detect_text_column_autodetect(self):
        column = detect_text_column(["id", "tweet", "label"])
        self.assertEqual(column, "tweet")

    def test_parse_csv_with_selected_text_column(self):
        csv_content = "col_a,body\n1,hello world\n2,bad weather\n"
        upload = SimpleUploadedFile(
            "sample.csv",
            csv_content.encode("utf-8"),
            content_type="text/csv",
        )
        texts, used_column, source_rows, source_columns = parse_uploaded_file(upload, selected_text_column="body")
        self.assertEqual(used_column, "body")
        self.assertEqual(texts, ["hello world", "bad weather"])
        self.assertEqual(source_columns, ["col_a", "body"])
        self.assertEqual(source_rows[0]["col_a"], "1")
        self.assertEqual(source_rows[0]["body"], "hello world")

    def test_parse_csv_raises_when_no_supported_column(self):
        csv_content = "id,message\n1,abc\n2,def\n"
        upload = SimpleUploadedFile(
            "sample.csv",
            csv_content.encode("utf-8"),
            content_type="text/csv",
        )
        with self.assertRaises(FileValidationError):
            parse_uploaded_file(upload)

    def test_read_csv_page_supports_pagination(self):
        predictions = []
        for idx in range(25):
            predictions.append(
                {
                    "text": f"tweet-{idx}",
                    "knn_label": "Positive",
                    "knn_score": 0.9,
                    "svm_label": "Negative",
                    "svm_score": 0.1,
                }
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(MEDIA_ROOT=tmp_dir):
                filename = generate_classification_csv(predictions, prefix="paging")
                rows, total_rows, current_page, total_pages = read_csv_page(filename, page=2, per_page=10)

        self.assertEqual(total_rows, 25)
        self.assertEqual(current_page, 2)
        self.assertEqual(total_pages, 3)
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0]["text"], "tweet-10")

    def test_read_csv_page_clamps_page_out_of_range(self):
        predictions = [
            {
                "text": "sample",
                "knn_label": "Positive",
                "knn_score": 0.9,
                "svm_label": "Positive",
                "svm_score": 0.9,
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(MEDIA_ROOT=tmp_dir):
                filename = generate_classification_csv(predictions, prefix="paging")
                rows, _, current_page, total_pages = read_csv_page(filename, page=99, per_page=10)

        self.assertEqual(total_pages, 1)
        self.assertEqual(current_page, 1)
        self.assertEqual(len(rows), 1)
