from django.urls import path # type: ignore
from .import views
urlpatterns = [
    path('',views.home,name='home'),
    path('input',views.input,name='input'),
    path('output',views.output,name='output'),
    path("blood-input/", views.blood_input, name="blood_input"),
]
