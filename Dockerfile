FROM python:3.8

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app app
COPY wsgi.py .

EXPOSE 5000
ENTRYPOINT uwsgi --wsgi-file wsgi.py --http :5000 --enable-threads
