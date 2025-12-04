FROM python:3.10-slim

RUN apt update -y
WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

# 1. Set a fixed location for the model cache inside the image
ENV HF_HOME=/app/hf_cache

# 2. Run a python one-liner to download and save the model NOW.
# This saves the model files into the Docker image permanently.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ====================================================================

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]