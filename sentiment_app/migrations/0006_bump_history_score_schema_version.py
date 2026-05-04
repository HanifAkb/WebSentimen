from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("sentiment_app", "0005_history_score_schema_version"),
    ]

    operations = [
        migrations.AlterField(
            model_name="predictionhistory",
            name="score_schema_version",
            field=models.PositiveSmallIntegerField(default=3),
        ),
        migrations.AlterField(
            model_name="scrapehistory",
            name="score_schema_version",
            field=models.PositiveSmallIntegerField(default=3),
        ),
    ]
