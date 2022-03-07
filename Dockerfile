FROM python:3.8

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./app ./app
EXPOSE 5000
ENTRYPOINT python3 -m app.main
