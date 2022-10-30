from django.shortcuts import render 
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def index(request):
    return render(request, 'index.html')

@csrf_exempt
def about(request):
    return render(request, 'about.html')

@csrf_exempt
def model(request):
    return render(request, 'model.html')

