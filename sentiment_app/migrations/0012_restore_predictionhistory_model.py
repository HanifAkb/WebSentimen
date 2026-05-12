from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("sentiment_app", "0011_rename_predictionhistory_classificationhistory"),
    ]

    operations = [
        migrations.RenameModel(
            old_name="ClassificationHistory",
            new_name="PredictionHistory",
        ),
        migrations.AlterField(
            model_name="predictionhistory",
            name="user",
            field=models.ForeignKey(
                on_delete=models.deletion.CASCADE,
                related_name="prediction_histories",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
    ]
