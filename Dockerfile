
FROM python:3.11

WORKDIR /genai_docker

RUN apt update -y && apt upgrade -y
RUN apt-get update -y

RUN pip3 install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install gradio==5.5.0

COPY . .

CMD uvicorn --host 0.0.0.0 --port 7861 gradio_ui:app --workers 4 
