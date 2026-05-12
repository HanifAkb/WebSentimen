from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("sentiment_app", "0010_bump_history_score_schema_version_for_neutral_threshold"),
    ]

    operations = [
        migrations.RenameModel(
            old_name="PredictionHistory",
            new_name="ClassificationHistory",
        ),
        migrations.AlterField(
            model_name="classificationhistory",
            name="user",
            field=models.ForeignKey(
                on_delete=models.deletion.CASCADE,
                related_name="classification_histories",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
    ]
