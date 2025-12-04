# AI-Librarian
End-to-end machine learning book recommendation engine designed to retrieve highly relevant books based on user queries, semantic preferences, category constraints, and emotional tone.


## Workflows

1. Update config.yaml
2. Update params.yaml
3. Update entity
4. Update the configuration manager in src config
5. Update the components
6. Update the pipeline
7. Update the main.py
8. Update the app.py



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
