python -m venv venv
.\venv\Scripts\activate
pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
pip install -r requirements.txt
python pip -m install -r requirements.txt
python manage.py makemigrations
python manage.py migrate
python manage.py runserver