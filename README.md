python -m venv venv

.\venv\Scripts\activate

pip install -r requirements.txt

pip install "modelscope[nlp]==0.4.7" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

pip install git+https://github.com/nateshmbhat/pyttsx3

python manage.py makemigrations

python manage.py migrate

python manage.py runserver
