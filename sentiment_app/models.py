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
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.user} | {self.start_date} - {self.end_date} ({self.tweet_count})"
