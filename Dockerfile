FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY artifacts /app/artifacts
COPY app.py /app/app.py

RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]