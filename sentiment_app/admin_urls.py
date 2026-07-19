from django.urls import path

from .views import (
    admin_delete_selected_history_view,
    admin_dashboard_view,
    admin_model_create_view,
    admin_model_delete_view,
    admin_model_edit_view,
    admin_prediction_history_delete_view,
    admin_prediction_history_edit_view,
    admin_scrape_history_delete_view,
    admin_scrape_history_edit_view,
    admin_user_create_view,
    admin_user_delete_view,
    admin_user_edit_view,
)

app_name = "admin"

urlpatterns = [
    path("", admin_dashboard_view, name="index"),
    path("models/add/", admin_model_create_view, name="model_add"),
    path("models/<path:version_name>/edit/", admin_model_edit_view, name="model_edit"),
    path("models/<path:version_name>/delete/", admin_model_delete_view, name="model_delete"),
    path("users/add/", admin_user_create_view, name="user_add"),
    path("users/<int:user_id>/edit/", admin_user_edit_view, name="user_edit"),
    path("users/<int:user_id>/delete/", admin_user_delete_view, name="user_delete"),
    path(
        "prediction-history/<int:history_id>/edit/",
        admin_prediction_history_edit_view,
        name="prediction_history_edit",
    ),
    path(
        "prediction-history/<int:history_id>/delete/",
        admin_prediction_history_delete_view,
        name="prediction_history_delete",
    ),
    path("history/delete-selected/", admin_delete_selected_history_view, name="history_delete_selected"),
    path("scrape-history/<int:history_id>/edit/", admin_scrape_history_edit_view, name="scrape_history_edit"),
    path(
        "scrape-history/<int:history_id>/delete/",
        admin_scrape_history_delete_view,
        name="scrape_history_delete",
    ),
]
