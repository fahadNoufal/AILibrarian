FROM python:3.10-slim

WORKDIR /app

# 1. Install System Dependencies & Clean up
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Install CPU-only PyTorch (Heavy, so we do it first to cache it)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install Requirements (Dependencies only, NOT your app)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Bake Model into Image
ENV HF_HOME=/app/hf_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 5. Copy Application Code
COPY . .

# We do this AFTER copying the code, so setup.py actually exists now.
# We use "." (install) instead of "-e ." (editable) because Docker doesn't need edit mode.
RUN pip install .

# 6. Start App
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]