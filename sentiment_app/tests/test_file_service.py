from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase

from sentiment_app.services.file_service import (
    FileValidationError,
    detect_text_column,
    parse_uploaded_file,
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
