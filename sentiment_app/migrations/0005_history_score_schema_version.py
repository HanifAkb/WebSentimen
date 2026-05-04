from django.db import migrations, models


def mark_existing_histories_as_legacy(apps, schema_editor):
    PredictionHistory = apps.get_model("sentiment_app", "PredictionHistory")
    ScrapeHistory = apps.get_model("sentiment_app", "ScrapeHistory")
    PredictionHistory.objects.all().update(score_schema_version=1)
    ScrapeHistory.objects.all().update(score_schema_version=1)


def noop_reverse(apps, schema_editor):
    return None


class Migration(migrations.Migration):
    dependencies = [
        ("sentiment_app", "0004_scrapehistory_resume_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="predictionhistory",
            name="score_schema_version",
            field=models.PositiveSmallIntegerField(default=2),
        ),
        migrations.AddField(
            model_name="scrapehistory",
            name="score_schema_version",
            field=models.PositiveSmallIntegerField(default=2),
        ),
        migrations.RunPython(mark_existing_histories_as_legacy, noop_reverse),
    ]
