from AILibrarian.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from AILibrarian.pipeline.stage_02_data_preprocessing import DataPreprocessingPipeline
from AILibrarian.pipeline.stage_03_embedding_generation_and_vector_storage import EmbeddingGenerationAndVectorStoragePipeline
from AILibrarian.pipeline.stage_04_data_classification import DataClassificationPipeline
from AILibrarian.pipeline.stage_05_sentimant_analysis import SentimantAnalysisPipeline


class ModelPipelines:
    def __init__(self):
        pass
    def run_all(self):
        
        data_ingestion_pipeline = DataIngestionPipeline()
        data_path = data_ingestion_pipeline.main()

        data_preprocessing_pipeline = DataPreprocessingPipeline(data_path)
        preprocessed_data = data_preprocessing_pipeline.main()

        embedding_generation_and_vector_storage_pipeline = EmbeddingGenerationAndVectorStoragePipeline()
        model,books_db = embedding_generation_and_vector_storage_pipeline.main()

        data_classification_pipeline = DataClassificationPipeline(preprocessed_data)
        classified_data = data_classification_pipeline.main()

        sentimant_analysis_pipeline = SentimantAnalysisPipeline(classified_data)
        final_dataset = sentimant_analysis_pipeline.main()
        
        return {
            'embedding_model':model,
            'db':books_db,
            'dataset':final_dataset
        }



