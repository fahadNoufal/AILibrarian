from AILibrarian.constants import ROOT_DIR
from AILibrarian.logger.custom_logger import logger
from AILibrarian.entity import DataRetrivalConfig
from pathlib import Path
import pandas as pd


class DataRetrival:
    def __init__(self,config:DataRetrivalConfig,root_dir=ROOT_DIR):
        self.config = config
        self.root_dir = root_dir
        
    def initialize_model_and_chroma_db(self,model,db):
        logger.info("Initializing Model and Chroma DB for data retrival...")
        self.model = model
        self.db = db
        
    def get_data_for_retrival(self,dataset:Path):
        logger.info("loading for data retrival...")
        df = pd.read_csv(Path(self.root_dir,dataset))
        return df
        
        
    def perform_similarity_search(self,query,df:pd.DataFrame)->pd.DataFrame:
        logger.info("Performing Similarity Search for data retrival...")
        
        query_vec = self.model.encode([query],normalize_embeddings=True)[0]
        recommendations = self.db._collection.query(
            query_embeddings=[query_vec],
            n_results=self.config.query_n_results
        )
        ids = list(int(i[1:14]) for i in recommendations['documents'][0])
        return df[df['isbn13'].isin(ids)]
    
    def filter_category(self,cat:str,df:pd.DataFrame)->pd.DataFrame:
        
        logger.info("Performing Category Filteration for data retrival...")
        
        if cat!= 'All':
            return df[df['simple_categories']==cat].head(self.config.return_n_results)
        else:
            return df.head(self.config.return_n_results)
        
    def sort_by_sentimance(self,sentimance:str,df:pd.DataFrame)-> pd.DataFrame:
        
        logger.info("Sorting Data By Sentimance for data retrival...")
        
        if sentimance =='All':
            return df
        elif sentimance == 'Happy':
            return df.sort_values(by='joy',ascending=False)
        elif sentimance == 'Surprising':
            return df.sort_values(by='surprise',ascending=False)
        elif sentimance == 'Angry':
            return df.sort_values(by='anger',ascending=False)
        elif sentimance == 'Suspenseful':
            return df.sort_values(by='fear',ascending=False)
        elif sentimance == 'Sad':
            return df.sort_values(by='sadness',ascending=False)