from django.urls import path

from .views import (
    beranda_view,
    download_output_view,
    history_detail_view,
    history_list_view,
    login_view,
    logout_view,
    predict_view,
    prediction_history_detail_view,
    resume_scrape_view,
    register_user_view,
    twitter_fetch_view,
)

urlpatterns = [
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),
    path("register/", register_user_view, name="register_user"),
    path("", beranda_view, name="home"),
    path("predict/", predict_view, name="predict"),
    path("scraping/", twitter_fetch_view, name="twitter_fetch"),
    path("history/", history_list_view, name="history_list"),
    path("history/<int:history_id>/", history_detail_view, name="history_detail"),
    path("history/<int:history_id>/resume/", resume_scrape_view, name="resume_scrape"),
    path("history/predict/<int:history_id>/", prediction_history_detail_view, name="prediction_history_detail"),
    path("download/<str:filename>/", download_output_view, name="download_output"),
]
