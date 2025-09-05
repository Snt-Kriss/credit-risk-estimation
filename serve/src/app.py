from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from fastapi_cache import FastAPICache
import pandas as pd
import pprint
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from typing import Dict
from serve.src.helpers.request import ClassifierRequest
from serve.src.registry.mlflow.handler import MlFlowHandler
from typing import List

import logging
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 
logging.basicConfig(format = log_format, level = logging.INFO)

app= FastAPI(title= 'Credit Risk Estimator API')

def get_handler():
    return MlFlowHandler()

@app.on_event("startup")
async def on_startup():
    FastAPICache.init(InMemoryBackend(), prefix='credit-risk-cache')

@app.get("/health")
@cache(expire=30)
def health(handler: MlFlowHandler= Depends(get_handler)) -> Dict:
    result= handler.check_mlflow_health()
    return {"status": result}

@app.post("/predict")
@cache(expire=60)
def predict(req: ClassifierRequest, handler: MlFlowHandler= Depends(get_handler))-> Dict:
    model= handler.get_production_model()

    df= pd.DataFrame([req.dict()])

    y_pred= model.predict(df)[0]

    return {"prediction": int(y_pred)}