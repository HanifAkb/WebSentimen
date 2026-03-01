from django.contrib import admin

from .models import PredictionHistory, ScrapeHistory


@admin.register(ScrapeHistory)
class ScrapeHistoryAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "query", "start_date", "end_date", "tweet_count", "created_at")
    search_fields = ("user__username", "query")
    list_filter = ("start_date", "end_date", "created_at")


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "input_type",
        "source_name",
        "sample_count",
        "created_at",
    )
    search_fields = ("user__username", "text_input", "source_name")
    list_filter = ("input_type", "created_at")
