FROM tensorflow/tensorflow:2.19.0-gpu

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY ./api /app/api
COPY ./models /app/models

ENV RUNNING_IN_DOCKER true

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
