from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('ofa', views.ofa, name='ofa'),
    path('caption', views.caption, name='caption'),
]

