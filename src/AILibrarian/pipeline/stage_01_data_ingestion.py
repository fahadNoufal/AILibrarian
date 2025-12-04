from pathlib import Path
from AILibrarian.config.configuration import ConfigurationManager
from AILibrarian.components.data_ingestion import DataIngestion
from AILibrarian.logger.custom_logger import logger

class DataIngestionPipeline:
    def __init__(self):
        logger.info("Data Ingestion Pipeline Started...")
        self.config = ConfigurationManager().get_data_ingestion_config()
        
    def main(self)->Path:
        data_ingestion = DataIngestion(self.config)
        output_data_path = data_ingestion.load_data()
        logger.info("Data Ingestion Pipeline Completed...")
        return output_data_path