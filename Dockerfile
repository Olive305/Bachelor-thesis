FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "/app/Data/IoT enriched event log paper/autoencoder.py"]
