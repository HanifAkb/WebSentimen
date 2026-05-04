from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("sentiment_app", "0008_bump_history_score_schema_version_for_probability_split"),
    ]

    operations = [
        migrations.AlterField(
            model_name="predictionhistory",
            name="score_schema_version",
            field=models.PositiveSmallIntegerField(default=6),
        ),
        migrations.AlterField(
            model_name="scrapehistory",
            name="score_schema_version",
            field=models.PositiveSmallIntegerField(default=6),
        ),
    ]
