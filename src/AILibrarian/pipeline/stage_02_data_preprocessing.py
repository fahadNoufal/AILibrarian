from pathlib import Path
from AILibrarian.logger.custom_logger import logger
from AILibrarian.config.configuration import ConfigurationManager
from AILibrarian.components.data_preprocessing import DataPreprocessing



class DataPreprocessingPipeline:
    def __init__(self,input_dataset:Path):
        logger.info("Data Preprocessing Pipeline Started...")
        self.config = ConfigurationManager().get_data_preprocessing_config()
        self.input_dataset = input_dataset
        
        
    def main(self)->Path:
        data_preprocessor = DataPreprocessing(self.config)
        output_data_path = data_preprocessor.get_preprocessed_data(self.input_dataset)
        logger.info("Data Preprocessing Pipeline Completed...")

        return output_data_path