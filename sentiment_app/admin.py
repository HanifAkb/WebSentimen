from django.contrib import admin

from .models import ScrapeHistory


@admin.register(ScrapeHistory)
class ScrapeHistoryAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "query", "start_date", "end_date", "tweet_count", "created_at")
    search_fields = ("user__username", "query")
    list_filter = ("start_date", "end_date", "created_at")
