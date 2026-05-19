FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Service account JSON inside the container
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/service-account.json
ENV BQ_SERVICE_ACCOUNT=/app/service-account.json
ENV GCS_SERVICE_ACCOUNT=/app/service-account.json
ENV GOOGLE_SERVICE_KEY=/app/service-account.json

WORKDIR /app

# System deps for psycopg2 (AlloyDB)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything including service-account.json
COPY . .

ENTRYPOINT ["python", "run_all.py"]
