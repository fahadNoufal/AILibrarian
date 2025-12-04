FROM python:3.10-slim

RUN apt update -y
WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]



gcloud artifacts repositories add-iam-policy-binding ailibrarian-repo \
    --location=us-central1 \
    --member="serviceAccount:ailibrarian@cogent-cocoa-478714-a9.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.writer"