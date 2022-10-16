from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('caption', views.caption, name='caption'),
]

