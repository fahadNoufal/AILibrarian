from pathlib import Path
from AILibrarian.logger.custom_logger import logger
from AILibrarian.components.sentimant_analysis import SentimantAnalysis
from AILibrarian.config.configuration import ConfigurationManager

class SentimantAnalysisPipeline:
    def __init__(self,input_dataset:Path):
        logger.info("Sentimant Analysis Pipeline Started...")

        self.config = ConfigurationManager().get_sentimant_analysis_config()
        self.input_dataset = input_dataset
        
    def main(self):
        sentimant_analysis = SentimantAnalysis(self.config)
        output_data_path = sentimant_analysis.get_sentimant_analysed_dataset(self.input_dataset)
        
        logger.info("Sentimant Analysis Pipeline Completed...")
        
        return output_data_path