from pathlib import Path

from django.conf import settings
from django.db import models
from django.utils.deconstruct import deconstructible
from django.utils.text import slugify


@deconstructible
class SentimentModelUploadTo:
    def __init__(self, kind: str):
        self.kind = str(kind or "").strip() or "model"

    def __call__(self, instance: "SentimentModelVersion", filename: str) -> str:
        return sentiment_model_storage_name(instance.version_name, self.kind, filename)


def sentiment_model_storage_name(version_name: str, kind: str, filename: str = "") -> str:
    extension = Path(str(filename or "")).suffix.lower() or ".joblib"
    version_slug = slugify(version_name or "model") or "model"
    file_stem = "knn_model" if str(kind or "").strip().lower() == "knn" else "svm_model"
    return f"sentiment_models/{version_slug}/{file_stem}{extension}"


def _sentiment_model_upload_to(kind: str):
    return SentimentModelUploadTo(kind)


class SentimentModelVersion(models.Model):
    version_name = models.CharField(max_length=100, unique=True)
    knn_model_file = models.FileField(upload_to=_sentiment_model_upload_to("knn"))
    svm_model_file = models.FileField(upload_to=_sentiment_model_upload_to("svm"))
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["version_name", "id"]

    @property
    def knn_file_name(self) -> str:
        return Path(str(self.knn_model_file.name or "")).name

    @property
    def svm_file_name(self) -> str:
        return Path(str(self.svm_model_file.name or "")).name

    def __str__(self) -> str:
        return self.version_name


class ScrapeHistory(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="scrape_histories",
    )
    query = models.TextField()
    model_version = models.CharField(max_length=100, blank=True)
    language = models.CharField(max_length=10, blank=True)
    start_date = models.DateField()
    end_date = models.DateField()
    tweet_count = models.PositiveIntegerField(default=0)
    rows = models.JSONField(default=list)
    is_complete = models.BooleanField(default=True)
    is_processing = models.BooleanField(default=False)
    resume_next_date = models.DateField(null=True, blank=True)
    stop_reason = models.CharField(max_length=32, blank=True)
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.user} | {self.start_date} - {self.end_date} ({self.tweet_count})"


class ScrapeTempChunk(models.Model):
    history = models.ForeignKey(
        ScrapeHistory,
        on_delete=models.CASCADE,
        related_name="temp_chunks",
    )
    chunk_index = models.PositiveIntegerField(default=0)
    rows = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["chunk_index", "id"]

    def __str__(self) -> str:
        return f"TempChunk history={self.history_id} idx={self.chunk_index}"


class PredictionHistory(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="prediction_histories",
    )
    source_name = models.CharField(max_length=255, blank=True)
    model_version = models.CharField(max_length=100, blank=True)
    text_column = models.CharField(max_length=100, blank=True)
    sample_count = models.PositiveIntegerField(default=1)
    columns = models.JSONField(default=list, blank=True)
    rows = models.JSONField(default=list, blank=True)
    is_processing = models.BooleanField(default=False)
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    @property
    def file_type_label(self) -> str:
        filename = (self.source_name or "").lower()
        if filename.endswith(".txt"):
            return "TXT"
        if filename.endswith(".csv"):
            return "CSV"
        return "CSV"

    @property
    def file_type_badge_class(self) -> str:
        if self.file_type_label == "TXT":
            return "text-bg-info"
        return "text-bg-primary"

    def __str__(self) -> str:
        return f"{self.user} | CSV/TXT ({self.sample_count})"
