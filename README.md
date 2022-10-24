python -m venv venv

.\venv\Scripts\activate

pip install "modelscope[nlp]==0.4.7" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

pip install git+https://github.com/nateshmbhat/pyttsx3

pip install -r requirements.txt

python manage.py makemigrations

python manage.py migrate

python manage.py runserver
