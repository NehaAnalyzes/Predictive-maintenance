# =============================================================================
# PREDICTIVE MAINTENANCE — DOCKERFILE
# Build : docker build -t predictive-maintenance .
# Run   : docker run -p 8000:8000 predictive-maintenance
# =============================================================================

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]