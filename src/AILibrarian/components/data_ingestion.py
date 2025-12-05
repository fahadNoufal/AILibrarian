import kagglehub
import pandas as pd
from pathlib import Path

from AILibrarian.constants import ROOT_DIR
from AILibrarian.logger.custom_logger import logger
from AILibrarian.utils.common import create_directories
from AILibrarian.entity import DataIngestionConfig




class DataIngestion:
    def __init__(self,config:DataIngestionConfig,root_dir=ROOT_DIR):
        self.config = config
        self.root_dir = root_dir
        
    def load_data(self):
        if not Path(self.root_dir,self.config.output_dataset).exists():
            logger.info('Fetching Books Dataset...')
            path = kagglehub.dataset_download(self.config.data_set)
            df = pd.read_csv(f"{path}/books.csv")
            create_directories([Path(self.root_dir,self.config.data_dir)])
            df.to_csv(Path(self.root_dir,self.config.output_dataset),index=False)
        logger.info('Successfully Fetched Books Dataset...')
        return self.config.output_dataset