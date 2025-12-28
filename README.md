# AI-Librarian
End-to-end machine learning book recommendation engine designed to retrieve highly relevant books based on user queries, semantic preferences, category constraints, and emotional tone.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch)
![Docker](https://img.shields.io/badge/Docker-24.0-2496ED?style=for-the-badge&logo=docker)
![Google Cloud Run](https://img.shields.io/badge/Google_Cloud-Run-4285F4?style=for-the-badge&logo=google-cloud)

## 📖 Overview

AILibrarian is an advanced NLP-powered recommendation engine designed to retrieve books based on semantic meaning rather than simple keyword matching. By leveraging transformer-based architectures, the system analyzes the description of books to understand context, genre, and emotional tone.

The application allows users to query naturally (e.g., *"books about overcoming grief in a fantasy world"*) and filter results based on specific emotional tones (e.g., *Joy, Suspense*) and dynamically generated categories.



## 🏗 Architecture & Methodology

The system operates on a three-stage pipeline:

### 1. Vector Embedding & Retrieval (Semantic Search)
Instead of lexical search, we utilize dense vector retrieval.
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Mechanism:** Book descriptions are encoded into **384-dimensional vectors** and stored in **ChromaDB**.
- **Retrieval:** User queries are vectorized in real-time. The system calculates **Cosine Similarity** between the query vector and the stored document embeddings to find the nearest semantic neighbors.

### 2. Zero-Shot Classification (Categorization)
Books are categorized without a pre-defined labeled dataset using Zero-Shot Learning.
- **Model:** `facebook/bart-large-mnli`
- **Method:** The model treats categorization as a Natural Language Inference (NLI) task, determining the probability that a book's description implies a specific genre label (e.g., *Fiction, Biography, History*).

### 3. Sentiment Analysis (Tone Filtering)
To align with user mood preference, the system analyzes the emotional undertone of the narrative.
- **Model:** `j-hartmann/emotion-english-distilroberta-base`
- **Architecture:** A distilled RoBERTa model fine-tuned on six specific emotions (anger, disgust, fear, joy, neutral, sadness, surprise).

## 🛠 Tech Stack

- **Language:** Python 3.10
- **Framework:** FastAPI (Asynchronous web server)
- **ML Libraries:** PyTorch, Hugging Face Transformers, Sentence-Transformers
- **Vector Database:** ChromaDB
- **Containerization:** Docker (Multi-stage build optimized for CPU inference)
- **Deployment:** Google Cloud Run (Serverless) via GitHub Actions CI/CD

## 🚀 Installation & Local Setup

### Prerequisites
- Python 3.9+
- Docker (Optional)

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/AILibrarian.git](https://github.com/your-username/AILibrarian.git)
cd AILibrarian

```

### 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the Application

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Access the Swagger UI at `http://localhost:8080/docs`.

## 🐳 Docker Deployment

The application is containerized using a CPU-optimized approach (installing CPU-only PyTorch to reduce image size).

```bash
# Build the image
docker build -t ailibrarian .

# Run the container
docker run -p 8080:8080 ailibrarian
```

## 📡 API Endpoints

### `GET /api/get_books`

Main search endpoint.

**Parameters:**
- `query` (str): The user's search input.
- `category` (str, optional): Filter by genre (Zero-shot classified).
- `tone` (str, optional): Filter by emotional tone (Sentiment analysis).

**Example Request:**

```HTTP
GET /api/get_books?query=space%20exploration&category=Sci-Fi&tone=surprise
```

## 🔄 CI/CD Pipeline

This project uses **GitHub Actions** for automated deployment:

1. **Build:** Creates a Docker image based on `Dockerfile`.
    
2. **Push:** Pushes the artifact to **Google Artifact Registry**.
    
3. **Deploy:** Updates the **Google Cloud Run** service with the new revision.
    
4. **Cleanup:** Automatically manages storage by removing old image tags.


```
                         ┌──────────────────────────────────┐
                         │     1. DATA INGESTION LAYER      │
                         └──────────────────────────────────┘
                                       │
                                       ▼
                     ┌─────────────────────────────────────────┐
                     │  Raw Book Metadata (Descriptions, ISBN) │
                     └─────────────────────────────────────────┘
                                       │
                                       ▼
         ┌────────────────────────────────────────────────────────────────────┐
         │      Pre-Processing Pipeline                                       │
         │  - text cleaning (stopwords, noise removal)                        │
         │  - metadata normalization                                           │
         │  - dataframe structuring                                            │
         └────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                         
                         
┌──────────────────────────────────────────────────────────────────────────────┐
│                      2. SEMANTIC EMBEDDING PIPELINE                          │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                      ┌────────────────────────────────┐
                      │ MiniLM Embedding Model         │
                      │ sentence-transformers/         │
                      │ all-MiniLM-L6-v2               │
                      │ - 384-dim vectors              │
                      │ - 22M params                   │
                      │ - L2 normalized                │
                      └────────────────────────────────┘
                                       │
                                       ▼
          ┌──────────────────────────────────────────────────────────┐
          │ Dense Embeddings for Every Book Description (384 dim)    │
          └──────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                      
                      
┌──────────────────────────────────────────────────────────────────────────────┐
│               3. CHROMADB VECTOR INDEX CONSTRUCTION                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                       ┌──────────────────────────────────┐
                       │ Chroma Collection                │
                       │ - cosine similarity metric       │
                       │ - persistent storage             │
                       │ - stores ids, descriptions,      │
                       │   384-dim vectors                │
                       └──────────────────────────────────┘
                                       │
                                       ▼
                   ┌──────────────────────────────────────────────┐
                   │ Vector Database Ready for Query Search       │
                   └──────────────────────────────────────────────┘
                                       │
                                       ▼


┌──────────────────────────────────────────────────────────────────────────────┐
│          4. QUERY UNDERSTANDING & RETRIEVAL PIPELINE                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │ User Query (Natural Language Input)         │
                    └─────────────────────────────────────────────┘
                                       │
                                       ▼
               ┌───────────────────────────────────────────────────────┐
               │ MiniLM Embeddings for Query (384-dim)                  │
               └───────────────────────────────────────────────────────┘
                                       │
                                       ▼
                 ┌────────────────────────────────────────────────────┐
                 │ Cosine Similarity Search in ChromaDB               │
                 │ Returns top-k semantically nearest documents       │
                 └────────────────────────────────────────────────────┘
                                       │
                                       ▼


┌──────────────────────────────────────────────────────────────────────────────┐
│                          5. CATEGORY FILTERING                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
           ┌─────────────────────────────────────────────────────────────┐
           │ Zero-Shot Classifier: facebook/bart-large-mnli              │
           │ - 400M params                                               │
           │ - Multi-label classification                               │
           │ - Assigns global categories (fiction, non-fiction, etc.)   │
           └─────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                     ┌─────────────────────────────────────────────┐
                     │ Category-Filtered Book Candidates            │
                     └─────────────────────────────────────────────┘
                                       │
                                       ▼


┌──────────────────────────────────────────────────────────────────────────────┐
│                     6. SENTIMENT & TONE ANALYSIS                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
         ┌──────────────────────────────────────────────────────────┐
         │ Emotion Model: j-hartmann/emotion-english-distilroberta │
         │ - 82M params                                            │
         │ - Outputs emotion distribution (anger, joy, etc.)       │
         └──────────────────────────────────────────────────────────┘
                                       │
                                       ▼
               ┌───────────────────────────────────────────────────────┐
               │ Optional Filtering by Sentiment / Emotional Tone      │
               └───────────────────────────────────────────────────────┘
                                       │
                                       ▼


┌──────────────────────────────────────────────────────────────────────────────┐
│                    7. MULTI-STAGE RANKING PIPELINE                           │
└──────────────────────────────────────────────────────────────────────────────┐
                                       │
                                       ▼

            ┌────────────────────────────────────────────────────────────┐
            │ Final Ranking Logic                                       │
            │ - semantic similarity                                     │
            │ - category relevance                                      │
            │ - sentiment alignment                                     │
            │ Combined via weighted scoring model                       │
            └────────────────────────────────────────────────────────────┘
                                       │
                                       ▼

                   ┌─────────────────────────────────────────────┐
                   │ Final Personalized Book Recommendations     │
                   └─────────────────────────────────────────────┘

---


```


## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
