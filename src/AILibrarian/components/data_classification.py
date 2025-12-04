from AILibrarian.constants import ROOT_DIR
from AILibrarian.logger.custom_logger import logger
from AILibrarian.entity import DataClassificationConfig
from pathlib import Path
from transformers import pipeline
import numpy as np
import pandas as pd

class DataClassification:
    def __init__(self,config:DataClassificationConfig,root_dir=ROOT_DIR):
        self.config = config
        self.root_dir = root_dir
        
        
    def label_category(self,cat,labels,possible_labels,classifier):
        if cat in labels.keys():
            return labels[cat]
        else:
            pred = classifier(cat,possible_labels)
            return pred['labels'][int(np.argmax(pred['scores']))]
        
        
    def store_classified_data(self,df:pd.DataFrame,simple_category_dataset:Path)->Path:
        
        simple_categories_classified = pd.read_csv(Path(self.root_dir,simple_category_dataset),index_col=0)
        combined_df = df.merge(simple_categories_classified, on = 'isbn10', how='left')
        
        combined_df['simple_categories'] = combined_df['simple_label_prediction']
        combined_df = combined_df.drop(columns=['simple_label_prediction'])

        combined_df = combined_df[~combined_df['categories'].isna()]
        logger.info('Storing classified data...')
        combined_df.to_csv(Path(self.root_dir,self.config.output_dataset),index=False)
        return self.config.output_dataset
    
    
    def get_classified_dataset(self,dataset:Path)->Path:
        
        labels = self.config.labels
        possible_labels = self.config.possible_labels
        
        logger.info('Reading data for classification...')
        df = pd.read_csv(Path(self.root_dir,dataset))
        
        output_path = Path(self.root_dir,self.config.simple_categories_classified_dataset)
        if not output_path.exists():
            logger.info('Loading Model for Zero-Shot-Classification...')
            classifier = pipeline("zero-shot-classification",
                                model="facebook/bart-large-mnli")
            
            df['simple_categories'] = df['categories'].apply( lambda x: labels[x] if x in labels.keys() else np.nan)
            
            df['simple_label_prediction'] = df['categories'].apply(lambda x: self.label_category(x,labels,possible_labels,classifier))
            simple_categories_classified = df[['isbn10','categories','simple_label_prediction']]
            simple_categories_classified.to_csv(Path(self.root_dir,self.config.simple_categories_classified_dataset),index=False)
        logger.info('Data Classified...')
        
        output_dataset = self.store_classified_data(df,self.config.simple_categories_classified_dataset)
        return output_dataset