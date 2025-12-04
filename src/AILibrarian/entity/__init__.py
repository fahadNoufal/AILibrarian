from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    data_dir: Path
    data_set: Path
    output_dataset: Path
    

@dataclass 
class DataPreprocessingConfig:
    min_description_len: int
    max_description_len: int
    output_dataset: Path
  

@dataclass
class EmbeddingGenerationAndVectorStorageConfig:
    chunk_size: int
    chunk_overlap: int
    tagged_description: Path
    seperator: str
    model: str
    embedding_data_path: Path
    collection_name: str
  

@dataclass
class DataClassificationConfig:
    pipeline_name: str
    model: str
    labels: dict
    possible_labels: list
    simple_categories_classified_dataset: Path
    output_dataset: Path

@dataclass
class SentimantAnalysisConfig:
  pipeline_name: str
  model: str
  emotions: list
  output_dataset: Path
  
@dataclass
class DataRetrivalConfig:
    query_n_results:int
    return_n_results:int