from django.db import migrations, models


def _column_names(schema_editor, table_name: str) -> set[str]:
    connection = schema_editor.connection
    with connection.cursor() as cursor:
        table_description = connection.introspection.get_table_description(cursor, table_name)
    return {column.name for column in table_description}


def _drop_column_if_exists(schema_editor, table_name: str, column_name: str) -> None:
    if column_name not in _column_names(schema_editor, table_name):
        return

    quoted_table = schema_editor.quote_name(table_name)
    quoted_column = schema_editor.quote_name(column_name)
    try:
        schema_editor.execute(f"ALTER TABLE {quoted_table} DROP COLUMN {quoted_column}")
    except Exception:
        pass


def repair_existing_history_schema(apps, schema_editor):
    ScrapeHistory = apps.get_model("sentiment_app", "ScrapeHistory")
    PredictionHistory = apps.get_model("sentiment_app", "PredictionHistory")
    scrape_table = ScrapeHistory._meta.db_table
    prediction_table = PredictionHistory._meta.db_table

    scrape_columns = _column_names(schema_editor, scrape_table)
    quoted_scrape_table = schema_editor.quote_name(scrape_table)
    quoted_resume_interval = schema_editor.quote_name("resume_interval_days")
    quoted_window_days = schema_editor.quote_name("window_days")

    if "resume_interval_days" not in scrape_columns and "window_days" in scrape_columns:
        schema_editor.execute(
            f"ALTER TABLE {quoted_scrape_table} RENAME COLUMN {quoted_window_days} TO {quoted_resume_interval}"
        )
    elif "resume_interval_days" not in scrape_columns:
        field = models.PositiveSmallIntegerField(default=1)
        field.set_attributes_from_name("resume_interval_days")
        schema_editor.add_field(ScrapeHistory, field)
    elif "window_days" in scrape_columns:
        schema_editor.execute(
            f"UPDATE {quoted_scrape_table} SET {quoted_resume_interval} = {quoted_window_days}"
        )
        _drop_column_if_exists(schema_editor, scrape_table, "window_days")

    _drop_column_if_exists(schema_editor, scrape_table, "score_schema_version")

    prediction_columns = _column_names(schema_editor, prediction_table)
    if "input_type" in prediction_columns:
        quoted_prediction_table = schema_editor.quote_name(prediction_table)
        quoted_input_type = schema_editor.quote_name("input_type")
        schema_editor.execute(
            f"DELETE FROM {quoted_prediction_table} WHERE {quoted_input_type} = %s",
            ["single"],
        )
    for column_name in ("input_type", "text_input", "output_filename", "score_schema_version"):
        _drop_column_if_exists(schema_editor, prediction_table, column_name)


class Migration(migrations.Migration):
    dependencies = [
        ("sentiment_app", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(repair_existing_history_schema, migrations.RunPython.noop),
    ]
