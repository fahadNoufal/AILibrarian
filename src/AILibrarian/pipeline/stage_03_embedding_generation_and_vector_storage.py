from AILibrarian.logger.custom_logger import logger
from AILibrarian.components.embedding_generation_and_vector_storage import EmbeddingGenerationAndVectorStorage
from AILibrarian.config.configuration import ConfigurationManager


class EmbeddingGenerationAndVectorStoragePipeline:
    def __init__(self):
        logger.info("Embedding Generation And Vector Storage Pipeline Pipeline Started...")
        
        self.config = ConfigurationManager().get_embedding_generation_and_vector_storage_config()
        
    def main(self):
        
        embedding_generation_and_vector_storage = EmbeddingGenerationAndVectorStorage(self.config)
        
        model,book_db = embedding_generation_and_vector_storage.get_vector_db()
        
        logger.info("Embedding Generation And Vector Storage Pipeline Completed...")
        
        return model,book_db