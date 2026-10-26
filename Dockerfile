FROM python:3.8-slim-bullseye

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app

CMD ["python3", "app.py"]