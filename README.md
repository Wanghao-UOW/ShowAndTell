
## Installtion:
#### Download Tiny (Distilled) Model from below Google Drive manually and copy to [OFA](https://github.com/Wanghao-UOW/ShowAndTell/tree/main/OFA) folder 
https://drive.google.com/drive/folders/1jpP-KNabBPAyOoDkj_U57GkpR4JBjMV0?usp=sharing

OR 

#### Download Large Model from below Google Drive manually and copy to [OFA](https://github.com/Wanghao-UOW/ShowAndTell/tree/main/OFA) folder

https://drive.google.com/drive/folders/1p__3PShIX6KbKBIyERSX9jrsUEN1hoR5?usp=sharing

###### Tiny is faster with reasonable accuracy of text caption while large is much slower with much higher accuracy of text caption

## Option 1 (using existing virtual environment): 

#### Download venv below and unzip to ShowAndTell folder
https://drive.google.com/file/d/1fTqD683dpdwtGnUMyQc3tLraZK7Ymj2G/view?usp=sharing

#### Run below commands:
    .\venv\Scripts\activate
    python manage.py runserver


## Option 2 (manually install packages): 
    ##### Windows 
    python -m venv venv
    .\venv\Scripts\activate
    python -m pip install --upgrade pip==22.3
    pip install -r requirements.txt
    pip install "modelscope[nlp]==0.4.7" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
    pip install git+https://github.com/nateshmbhat/pyttsx3
    python manage.py runserver 
    
    ###### For public facing run  
    ###### python manage.py runserver 0.0.0.0:80
    
    ##### Ubuntu 
    sudo python3 -m pip install --upgrade pip==22.3
    sudo pip install -r requirements.txt
    sudo pip install "modelscope[nlp]==0.4.7" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
    sudo apt update && sudo apt install espeak ffmpeg libespeak1
    sudo pip install git+https://github.com/nateshmbhat/pyttsx3
    sudo python3 manage.py runserver
    
    ###### For public facing run 
    ###### python3 manage.py runserver 0.0.0.0:80
