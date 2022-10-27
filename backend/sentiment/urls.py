from django.urls import path
from . import views

urlpatterns = [
    path("sentence", views.eval_sentence, name="sentence"),
    path("file", views.eval_file, name="file"),
]
