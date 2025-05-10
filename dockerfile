FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

COPY requirements.txt
COPY ./api /app/api
COPY ./models /app/models

ENV RUNNING_IN_DOCKER true


RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
