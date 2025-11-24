from django.urls import path
from . import views  # import views from the same app

urlpatterns = [
    path('', views.landing_page, name='landing'),       # Landing page (home)
    path('predict/', views.predict_churn, name='predict_churn'),  # Prediction form
]
