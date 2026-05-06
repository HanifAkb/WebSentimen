from django.conf import settings
from django.db import models


class ScrapeHistory(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="scrape_histories",
    )
    query = models.TextField()
    language = models.CharField(max_length=10, blank=True)
    start_date = models.DateField()
    end_date = models.DateField()
    tweet_count = models.PositiveIntegerField(default=0)
    rows = models.JSONField(default=list)
    score_schema_version = models.PositiveSmallIntegerField(default=7)
    is_complete = models.BooleanField(default=True)
    resume_next_date = models.DateField(null=True, blank=True)
    stop_reason = models.CharField(max_length=32, blank=True)
    window_days = models.PositiveSmallIntegerField(default=1)
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
    class InputType(models.TextChoices):
        SINGLE = "single", "Kalimat"
        FILE = "file", "CSV/TXT"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="prediction_histories",
    )
    input_type = models.CharField(max_length=12, choices=InputType.choices)
    text_input = models.TextField(blank=True)
    source_name = models.CharField(max_length=255, blank=True)
    text_column = models.CharField(max_length=100, blank=True)
    sample_count = models.PositiveIntegerField(default=1)
    columns = models.JSONField(default=list, blank=True)
    rows = models.JSONField(default=list, blank=True)
    score_schema_version = models.PositiveSmallIntegerField(default=7)
    output_filename = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.user} | {self.get_input_type_display()} ({self.sample_count})"
