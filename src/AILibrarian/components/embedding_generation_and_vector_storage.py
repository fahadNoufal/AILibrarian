import pandas as pd
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

from AILibrarian.constants import ROOT_DIR
from AILibrarian.logger.custom_logger import logger
from AILibrarian.entity import EmbeddingGenerationAndVectorStorageConfig


class EmbeddingGenerationAndVectorStorage:
    def __init__(self,config:EmbeddingGenerationAndVectorStorageConfig,root_dir=ROOT_DIR):
        logger.info('Data Embedding Config Initialized...')
        self.config = config
        self.root_dir = root_dir
        self.model = None
        
    def get_embedding_model_dataset(self,docs):
        
        logger.info('Loading Model for Data Embedding...')
        
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        logger.info('Successfully loaded Model for Data Embedding...')
        
        output_path = Path(self.root_dir, self.config.embedding_data_path)
        if not output_path.exists():
            logger.info('Creating Data Embeddings...')
            
            embedds = self.model.encode([doc.page_content for doc in docs],
                                    normalize_embeddings=True,
                                    show_progress_bar=True
                                )
            pd.DataFrame(embedds).to_csv(Path(self.root_dir,self.config.embedding_data_path),index=False)
        logger.info('Data Embeddings Generated...')
            
        return (self.model,self.config.embedding_data_path)
        
    
    def get_vector_db(self):
        logger.info('Loading Tagged Description For Generating Embeddings...')        
        raw_docs = TextLoader(Path(self.root_dir,self.config.tagged_description)).load()
        
        text_splitter = CharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separator=self.config.seperator
            )
        
        logger.info('Splitting Tagged Description into chunks...')
        docs = text_splitter.split_documents(raw_docs)
        
        logger.info('Loading Model and Generating Embeddings...')
        model,embedding_data_path = self.get_embedding_model_dataset(docs)
        df = pd.read_csv(Path(self.root_dir,embedding_data_path))
        embds = df.values
        
        logger.info('Creating Chroma Vector DB...')
        book_db = Chroma(
            collection_name="books_collection",
            embedding_function=None,
        )

        logger.info('Storing Embeddings into Chorma DB collection...')
        book_db._collection.add(
            embeddings=embds,
            documents=[doc.page_content for doc in docs],
            ids=[f"doc_{i}" for i in range(len(docs))]
        )
        return (model,book_db)