FROM python:3.10-slim

WORKDIR /app

# 1. Install System Tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Install PyTorch (CPU version to save space)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install dependencies (Remove "-e ." from requirements.txt first!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Bake Model
ENV HF_HOME=/app/hf_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 5. Copy EVERYTHING (src, config, main.py) to /app
COPY . .

# Instead of installing the package and moving it away,
# we just add the source folder to Python's path.
# This keeps 'src' and 'config' side-by-side in /app.
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# 6. Start the App
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]