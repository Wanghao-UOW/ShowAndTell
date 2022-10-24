python -m venv venv

.\venv\Scripts\activate

pip install -r requirements.txt

pip install "modelscope[nlp]==0.4.7" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

Download Large Model from below Google Drive manually if you can't clone and overwrite the same folder in OFA
https://drive.google.com/drive/folders/1p__3PShIX6KbKBIyERSX9jrsUEN1hoR5?usp=sharing 

pip install git+https://github.com/nateshmbhat/pyttsx3

python manage.py makemigrations

python manage.py migrate

python manage.py runserver
