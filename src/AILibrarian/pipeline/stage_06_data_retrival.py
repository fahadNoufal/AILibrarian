from pathlib import Path
from AILibrarian.logger.custom_logger import logger
from AILibrarian.components.data_retrival import DataRetrival
import pandas as pd
from AILibrarian.config.configuration import ConfigurationManager

class DataRetrivalPipeline:
    def __init__(self,model,db,input_dataset:Path):
        logger.info("Data Retrival Pipeline Started...")
        self.model = model
        self.db = db
        self.config = ConfigurationManager().get_data_retrival_config()
        self.input_dataset = input_dataset
        
    def search(self,query:str,cat:str,tone:str) -> pd.DataFrame:
        model = self.model
        db = self.db
        data_retrival = DataRetrival(self.config)
        data_retrival.initialize_model_and_chroma_db(model=model,db=db)
        df = data_retrival.get_data_for_retrival(self.input_dataset)
        df = data_retrival.perform_similarity_search(query,df)
        df = data_retrival.filter_category(cat,df)
        df = data_retrival.sort_by_sentimance(tone,df)
        required_columns = [
            'isbn13',
            'title',
            'authors',
            'categories',
            'thumbnail',
            'description',
            'published_year',
            'average_rating',
            'num_pages',
            'ratings_count',
            'simple_categories',
            'title and subtitle'
            ]
        df = df[required_columns]
        
        logger.info("Data Retrival Pipeline Completed...")
        
        return df