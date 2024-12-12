from django.urls import path
from .views import upload_and_process

urlpatterns = [
    path('process/', upload_and_process, name='process'),
]
