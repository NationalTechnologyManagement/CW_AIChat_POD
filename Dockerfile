# CW AI Chat Pod — FastAPI service.
# Plain container, env-driven: runs identically on Railway, Azure Container Apps, or local compose.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Honors $PORT (Railway/Azure inject it); defaults to 8080 locally.
EXPOSE 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
