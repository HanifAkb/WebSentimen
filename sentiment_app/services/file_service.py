from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
from django.conf import settings

COMMON_TEXT_COLUMNS = ("text", "tweet", "content", "sentence")
ALLOWED_CONTENT_TYPES = {
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
    "text/plain",
    "application/octet-stream",
}


class FileValidationError(ValueError):
    pass


def _max_upload_size() -> int:
    return int(getattr(settings, "SENTIMENT_UPLOAD_MAX_SIZE", 5 * 1024 * 1024))


def _allowed_extensions() -> set[str]:
    configured = getattr(settings, "SENTIMENT_ALLOWED_UPLOAD_EXTENSIONS", {".csv", ".txt"})
    return {str(ext).lower() for ext in configured}


def validate_uploaded_file(uploaded_file: Any) -> str:
    if uploaded_file is None:
        raise FileValidationError("File belum dipilih.")

    extension = Path(uploaded_file.name).suffix.lower()
    if extension not in _allowed_extensions():
        raise FileValidationError("Jenis file tidak valid. Hanya CSV atau TXT yang diizinkan.")

    if uploaded_file.size > _max_upload_size():
        size_mb = _max_upload_size() / (1024 * 1024)
        raise FileValidationError(f"Ukuran file terlalu besar. Maksimal ukuran unggahan adalah {size_mb:.0f} MB.")

    content_type = (getattr(uploaded_file, "content_type", "") or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise FileValidationError("Content type file tidak valid.")

    return extension


def detect_text_column(columns: list[str], selected: str | None = None) -> str:
    normalized = {column.strip().lower(): column for column in columns}

    preferred = (selected or "").strip().lower()
    if preferred:
        if preferred in normalized:
            return normalized[preferred]
        raise FileValidationError(f"Kolom '{selected}' tidak ditemukan pada CSV.")

    for candidate in COMMON_TEXT_COLUMNS:
        if candidate in normalized:
            return normalized[candidate]

    raise FileValidationError(
        "Tidak ditemukan kolom teks yang bisa digunakan. Tambahkan salah satu kolom: text, tweet, content, sentence, "
        "atau isi nama kolom secara manual."
    )


def parse_uploaded_file(
    uploaded_file: Any, selected_text_column: str | None = None
) -> tuple[list[str], str, list[dict[str, str]], list[str]]:
    extension = validate_uploaded_file(uploaded_file)

    if extension == ".csv":
        return _parse_csv(uploaded_file, selected_text_column)
    if extension == ".txt":
        return _parse_txt(uploaded_file)

    raise FileValidationError("Ekstensi file tidak didukung.")


def _parse_csv(
    uploaded_file: Any, selected_text_column: str | None = None
) -> tuple[list[str], str, list[dict[str, str]], list[str]]:
    uploaded_file.seek(0)
    try:
        dataframe = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        dataframe = pd.read_csv(uploaded_file, encoding="latin-1")
    except Exception as exc:
        raise FileValidationError(f"Gagal membaca file CSV: {exc}") from exc

    if dataframe.empty:
        raise FileValidationError("File CSV yang diunggah kosong.")

    column_name = detect_text_column([str(column) for column in dataframe.columns], selected_text_column)
    source_columns = [str(column) for column in dataframe.columns]
    texts: list[str] = []
    source_rows: list[dict[str, str]] = []

    for _, data_row in dataframe.iterrows():
        text_value = str(data_row.get(column_name, "") if pd.notna(data_row.get(column_name, "")) else "").strip()
        if not text_value:
            continue

        normalized_row: dict[str, str] = {}
        for column in source_columns:
            value = data_row.get(column, "")
            normalized_row[column] = "" if pd.isna(value) else str(value)

        texts.append(text_value)
        source_rows.append(normalized_row)

    if not texts:
        raise FileValidationError("Tidak ada baris teks yang terisi pada kolom CSV terpilih.")

    return texts, column_name, source_rows, source_columns


def _parse_txt(uploaded_file: Any) -> tuple[list[str], str, list[dict[str, str]], list[str]]:
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="ignore")
    else:
        text = str(raw)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise FileValidationError("File TXT tidak memiliki baris teks yang terisi.")
    source_columns = ["text"]
    source_rows = [{"text": line} for line in lines]
    return lines, "text", source_rows, source_columns


def create_uploaded_file_for_tests(name: str, content: str, content_type: str = "text/plain") -> io.BytesIO:
    file_obj = io.BytesIO(content.encode("utf-8"))
    file_obj.name = name
    file_obj.content_type = content_type
    file_obj.size = len(content.encode("utf-8"))
    file_obj.seek(0)
    return file_obj
