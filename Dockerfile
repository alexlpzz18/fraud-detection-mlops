FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

EXPOSE 8080

CMD ["uvicorn", "src.api_inferencia:app", "--host", "0.0.0.0", "--port", "8080"]