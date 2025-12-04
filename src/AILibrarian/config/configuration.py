from AILibrarian.entity import DataIngestionConfig,DataPreprocessingConfig,EmbeddingGenerationAndVectorStorageConfig,DataClassificationConfig,SentimantAnalysisConfig,DataRetrivalConfig
from AILibrarian.constants import *
from AILibrarian.utils.common import create_directories,read_yaml

class ConfigurationManager:
    
    def __init__(
            self,
            config_path = CONFIG_FILE_PATH,
            params_path = PARAMS_FILE_PATH,
            root_dir = ROOT_DIR
            ):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        self.root_dir = root_dir
        
    
    def get_data_ingestion_config(self):
        
        config = self.config.data_ingestion
        data_ingestion_config = DataIngestionConfig(**config)
        return data_ingestion_config
    
    def get_data_preprocessing_config(self):
        
        config = self.config.data_preprocessing
        data_preprocessing_config = DataPreprocessingConfig(**config)
        return data_preprocessing_config
    
    def get_embedding_generation_and_vector_storage_config(self):
        
        config = self.config.embedding_generation_and_vector_storage
        embedding_generation_and_vector_storage_config = EmbeddingGenerationAndVectorStorageConfig(**config)
        return embedding_generation_and_vector_storage_config
    
    def get_data_classification_config(self):
        
        config = self.config.data_classification
        data_classification_config = DataClassificationConfig(**config)
        
        return data_classification_config
    
    def get_sentimant_analysis_config(self):
        config = self.config.sentimant_analysis
        sentimant_analysis_config = SentimantAnalysisConfig(**config)
        return sentimant_analysis_config
        
    def get_data_retrival_config(self):
        config = self.config.data_retrival
        data_retrival_config = DataRetrivalConfig(**config)
        return data_retrival_config