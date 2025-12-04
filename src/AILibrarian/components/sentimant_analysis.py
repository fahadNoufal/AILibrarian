from AILibrarian.constants import ROOT_DIR
from AILibrarian.logger.custom_logger import logger
from AILibrarian.entity import SentimantAnalysisConfig
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm
import pandas as pd


class SentimantAnalysis:
    def __init__(self,config:SentimantAnalysisConfig,root_dir=ROOT_DIR):
        self.config = config
        self.root_dir = root_dir
        
        
    def get_max_emotion(self,desc,classifier):
        sentences = desc.split('. ')
        predicted_emotions = classifier(sentences)
        sentence_emotion_scores = []

        for sentence_results in predicted_emotions:
            scores = {}
            for emotion_data in sentence_results:
                scores[emotion_data['label']] = emotion_data['score']
            sentence_emotion_scores.append(scores)

        emotions_df = pd.DataFrame(sentence_emotion_scores)
        return emotions_df.max()
    
    
    def get_sentimant_scores(self,df:pd.DataFrame):
        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
        emotions_df_columns = ['isbn10']+ list(self.config.emotions)
        emotions_df = pd.DataFrame(columns=emotions_df_columns)

        for i in tqdm(range(df.shape[0])):
            emotion_scores = self.get_max_emotion(df.loc[i,'description'],classifier)
            emotions_df.loc[i,emotion_scores.index] = emotion_scores.values
            emotions_df.loc[i,'isbn10'] = df.loc[i,'isbn10']
            
        
    def get_sentimant_analysed_dataset(self,input_datast:Path):
        df = pd.read_csv(Path(self.root_dir,input_datast))
        output_path = Path(self.root_dir,self.config.output_dataset)
        if not output_path.exists():
            logger.info('Loading Model for Sentimant Analysis...')
            
            # additional cleanup
            df['authors'] = df['authors'].fillna('Author Unknown')
            df['thumbnail'] = df['thumbnail'].fillna('thumbnail missing')
            df['thumbnail'] = df['thumbnail'] + '&fife=w800'
            df = df.drop(columns='simple_category')
            
            emotions_df = self.get_sentimant_scores(df)
            books_with_emotion_scores = df.merge(emotions_df,on='isbn10',how='left')
            books_with_emotion_scores.to_csv(Path(self.root_dir,self.config.output_dataset),index=False)
        logger.info('Sentimant Analyzed...')
        
        return self.config.output_dataset