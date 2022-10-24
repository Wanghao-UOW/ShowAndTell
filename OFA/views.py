from logging import exception
from django.shortcuts import render 
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

from django.shortcuts import render, redirect
import pyttsx3
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage 
 
from .forms import OFAImageForm 

import os
from datetime import datetime

# better accuracy but slower 
# img_captioning = pipeline(
#   Tasks.image_captioning, 
#   model='OFA/ofa_image-caption_coco_large_en')

# lower accuracy but very fast
img_captioning = pipeline(
    Tasks.image_captioning, 
    model='damo/ofa_image-caption_coco_distilled_en')

# result = img_captioning({'image': 'https://farm9.staticflickr.com/8044/8145592155_4a3e2e1747_z.jpg'})
# print(result[OutputKeys.CAPTION]) # 'a bunch of donuts on a wooden board with popsicle sticks'

@csrf_exempt
def index(request):
    return render(request, 'index.html')

@csrf_exempt
def about(request):
    return render(request, 'about.html')


caption = "No Caption!"
@csrf_exempt
def model(request):
    start_time = datetime.now()
    if request.method == 'POST': 

        # Remove all old images
        directory = "OFA/static"
        files_in_directory = os.listdir(directory)
        for file in files_in_directory:
            path_to_file = os.path.join(directory, file)
            os.remove(path_to_file)

        # Image Display
        form = OFAImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            img_object = form.instance 
            
            # OFA Image Caption from pre-trained
            result = img_captioning({'image': img_object.image.name})
            global caption
            caption = result[OutputKeys.CAPTION][0]

            # Save audio file
            engine = pyttsx3.init()
            engine.save_to_file(caption, 'OFA/audio.mp3')
            engine.runAndWait()

            # timer
            end_time = datetime.now()
            time_lapsed =  end_time - start_time

            # format time
            t = str(round(time_lapsed.total_seconds(),1))+'s'

            # data will send to frontend
            context = {
                'form': form, 
                'img_obj': img_object, 
                'result': result[OutputKeys.CAPTION][0], 
                'time_lapsed': t
            }
            return render(request, 'OFA.html', context)  
    else:
        form = OFAImageForm()
    
    # timer
    end_time = datetime.now()
    time_lapsed = end_time - start_time

    # format time
    t = str(round(time_lapsed.total_seconds(),1))+'s'

    # data will send to frontend
    context = {
        'form' : form,  
        'result': 'No valid Image!', 
        'time_lapsed': t
    }
    return render(request, 'OFA.html', context)

""" @csrf_exempt
def caption1(request):

    value = request.FILES.get("image") 
    filepath = 'OFA\\static\\' + request.FILES.get("image").name

    default_storage.save(filepath, ContentFile(value.read()))

    result = img_captioning({'image': filepath})
    context = {
        'caption': result[OutputKeys.CAPTION]
    }
    return render(request, 'OFA.html', context)
 """
    
