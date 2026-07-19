import django.db.models.deletion
import sentiment_app.models
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="ScrapeHistory",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("query", models.TextField()),
                ("model_version", models.CharField(blank=True, max_length=100)),
                ("language", models.CharField(blank=True, max_length=10)),
                ("start_date", models.DateField()),
                ("end_date", models.DateField()),
                ("tweet_count", models.PositiveIntegerField(default=0)),
                ("rows", models.JSONField(default=list)),
                ("is_complete", models.BooleanField(default=True)),
                ("is_processing", models.BooleanField(default=False)),
                ("resume_next_date", models.DateField(blank=True, null=True)),
                ("stop_reason", models.CharField(blank=True, max_length=32)),
                ("error_message", models.TextField(blank=True, default="")),
                ("resume_interval_days", models.PositiveSmallIntegerField(default=1)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="scrape_histories",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
        migrations.CreateModel(
            name="PredictionHistory",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("source_name", models.CharField(blank=True, max_length=255)),
                ("model_version", models.CharField(blank=True, max_length=100)),
                ("text_column", models.CharField(blank=True, max_length=100)),
                ("sample_count", models.PositiveIntegerField(default=1)),
                ("columns", models.JSONField(blank=True, default=list)),
                ("rows", models.JSONField(blank=True, default=list)),
                ("is_processing", models.BooleanField(default=False)),
                ("error_message", models.TextField(blank=True, default="")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="prediction_histories",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
        migrations.CreateModel(
            name="SentimentModelVersion",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("version_name", models.CharField(max_length=100, unique=True)),
                ("knn_model_file", models.FileField(upload_to=sentiment_app.models.SentimentModelUploadTo("knn"))),
                ("svm_model_file", models.FileField(upload_to=sentiment_app.models.SentimentModelUploadTo("svm"))),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "ordering": ["version_name", "id"],
            },
        ),
        migrations.CreateModel(
            name="ScrapeTempChunk",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("chunk_index", models.PositiveIntegerField(default=0)),
                ("rows", models.JSONField(default=list)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "history",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="temp_chunks",
                        to="sentiment_app.scrapehistory",
                    ),
                ),
            ],
            options={
                "ordering": ["chunk_index", "id"],
            },
        ),
    ]
