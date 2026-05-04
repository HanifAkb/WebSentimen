from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("sentiment_app", "0007_bump_history_score_schema_version_for_soft_voting"),
    ]

    operations = [
        migrations.AlterField(
            model_name="predictionhistory",
            name="score_schema_version",
            field=models.PositiveSmallIntegerField(default=5),
        ),
        migrations.AlterField(
            model_name="scrapehistory",
            name="score_schema_version",
            field=models.PositiveSmallIntegerField(default=5),
        ),
    ]
