from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("sentiment_app", "0002_repair_existing_history_schema"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="scrapehistory",
            name="resume_interval_days",
        ),
    ]
