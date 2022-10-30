from django.urls import path
from . import views


urlpatterns = [
    # path('', views.textaudio, name='show and tell'),
    path('', views.text2audio, name='text2audio')
]