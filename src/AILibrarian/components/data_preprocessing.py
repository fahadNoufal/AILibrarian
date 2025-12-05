import numpy as np
import pandas as pd
from pathlib import Path

from AILibrarian.constants import ROOT_DIR
from AILibrarian.logger.custom_logger import logger
from AILibrarian.entity import DataPreprocessingConfig

class DataPreprocessing:
    def __init__(self,config:DataPreprocessingConfig,root_dir=ROOT_DIR):
        self.config = config
        self.root_dir = root_dir
        
    def get_preprocessed_data(self,data_path):
        logger.info('Preprocessing the data...')
        output_path = Path(self.root_dir,self.config.output_dataset)
        if not output_path.exists():
            data_path = Path(self.root_dir,data_path)
            df = pd.read_csv(data_path)
            df['is_missing_description'] = np.where(df['description'].isna(),1,0)
            df['age_of_book'] = 2025 - df['published_year']
    
            df = df[
                ~(df['description'].isna()) &
                ~(df['num_pages'].isna()) &
                ~(df['average_rating'].isna()) &
                ~(df['published_year'].isna()) &
                ~(df['ratings_count'].isna()) 
            ]
            df['description_len'] = df.description.apply(lambda x: len(x.split()))
            df['desc_above_25_words'] = df.description_len.between(25,400).astype(int)
            desc_above_25_words = df[df.desc_above_25_words==1]
            
            desc_above_25_words['title and subtitle'] = np.where(
                desc_above_25_words['subtitle'].isna(),
                desc_above_25_words['title'],
                desc_above_25_words[['title','subtitle']].astype(str).agg(': '.join,axis=1)
            )
            
            desc_above_25_words['tagged_description'] = desc_above_25_words[['isbn13','description']].astype(str).agg(' '.join,axis=1)
            # desc_above_25_words[['tagged_description']].to_csv(Path(self.root_dir,'artifacts/datasets/tagged_descriptions'),index=False)
            cleaned_df = desc_above_25_words.drop(columns=['subtitle','is_missing_description','age_of_book','description_len','desc_above_25_words'])
            
            cleaned_df.to_csv(Path(self.root_dir,self.config.output_dataset),index=False)
        logger.info('Completed data preprocessing...')
    
        return self.config.output_dataset
        
        
