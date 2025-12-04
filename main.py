import gc
import torch
from contextlib import asynccontextmanager

from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware

from AILibrarian.logger.custom_logger import logger
# Model Pipelines
from app import ModelPipelines
#Retrival Pipelin
from AILibrarian.pipeline.stage_06_data_retrival import DataRetrivalPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    pipelines = ModelPipelines()
    model_resources = pipelines.run_all()
    
    logger.info("Successfully Ran Pipelines...")
    
    app.state.db = model_resources['db']
    
    app.state.data_retrival_pipeline = DataRetrivalPipeline(
        model = model_resources['embedding_model'],
        db = model_resources['db'],
        input_dataset = model_resources['dataset']
    )

    logger.info('App Loaded ------------------------------------')
    print('App Loaded ------------------------------------')

    yield
    
    if hasattr(app.state, "data_retrival_pipeline"):
        del app.state.data_retrival_pipeline

    if hasattr(app.state, "db"):
        del app.state.db

    gc.collect()
    torch.cuda.empty_cache()
    
    logger.info("App closed...")

app = FastAPI(lifespan=lifespan)

# Setting up FastAPI app with CORS middleware .
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/api/get_books/')
async def get_news(request:Request,query:str,category:str,tone:str):
    if query.strip() =='' or category.strip() =='' or tone.strip() =='':
        return {'message':'input in not valid'}
    logger.info("Sucessfuly recieved get request...")
    
    # if author is missing -> 'Author Unknown'
    # if thumbnail is missing -> 'thumbnail missing'
    
    data_retrival_pipeline = request.app.state.data_retrival_pipeline
    result = data_retrival_pipeline.search(query,category,tone)
    result = result.to_dict()
    
    logger.info("Sucessfully loaded result data...")

    return result