FROM continuumio/miniconda3

WORKDIR /home/app

COPY api.py .
COPY requirements.txt .

RUN pip install -r requirements.txt


CMD uvicorn api:app --port $PORT --host 0.0.0.0