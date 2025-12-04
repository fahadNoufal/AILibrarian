# 1. Base Image
FROM python:3.10-slim

WORKDIR /app

# 2. Install system tools and CLEAN UP immediately (Saves space)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# If we let requirements.txt do it, it downloads the GPU version (~4GB).
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Copy ONLY requirements first (Docker Layer Caching)
# This makes future builds faster. If you change code, we don't re-install libs.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Bake the model into the image
ENV HF_HOME=/app/hf_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 5. Copy the rest of the application code
COPY . .

# 6. Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]


