from pathlib import Path
from AILibrarian.logger.custom_logger import logger
from AILibrarian.components.data_classification import DataClassification
from AILibrarian.config.configuration import ConfigurationManager

class DataClassificationPipeline:
    def __init__(self,input_dataset:Path):
        logger.info("Data Classification Pipeline Started...")

        self.config = ConfigurationManager().get_data_classification_config()
        self.input_dataset = input_dataset
        
    def main(self):
        data_classificaiton = DataClassification(self.config)
        output_data_path = data_classificaiton.get_classified_dataset(self.input_dataset)
        
        logger.info("Data Classification Pipeline Completed...")
        
        return output_data_path
        