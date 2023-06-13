FROM python:3.10.3-slim-buster

WORKDIR /app

RUN apt-get update 

RUN apt-get install -y wget 

RUN wget https://storage.googleapis.com/freshcancoba1/model/model.h5

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

ENV HOST 0.0.0.0

EXPOSE 8080

CMD ["python", "main.py"]
