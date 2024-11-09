import json
from typing import Any

#i
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from fastapi import APIRouter, HTTPException,Response
from fastapi.encoders import jsonable_encoder
from patient_survival_prediction_model import __version__ as model_version
from patient_survival_prediction_model.predict import make_prediction
from patient_survival_prediction_model.processing.data_manager import load_dataset
from patient_survival_prediction_model.config.core import config
from schemas import PredictionResults
from app import __version__, schemas
from app.config import settings,r2_metric
import prometheus_client as prom

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs) -> Any:
    

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))
    
    if isinstance(results["predictions"], (list, np.ndarray)):
        prediction = results["predictions"][0]
    else:
        prediction = results["predictions"]
        
    return PredictionResults(
        errors=results.get("errors"),
        version=results.get("version", "unknown"),
        predictions=prediction)

# Function for updating metrics
def update_metrics():
    test_data = load_dataset(file_name = config.app_config.training_data_file)
    test = test_data.sample(100)
    test_feat = test.drop('DEATH_EVENT', axis=1)
    test_cnt = test['DEATH_EVENT'].values
    test_pred = make_prediction(input_data=test_feat)['predictions']
    print(test_pred)
    _predictions = list(test_pred)
    _predictions = np.where(np.array(_predictions) == "The patient is predicted to not have a death event.", 0, 1)
    r2 = r2_score(test_cnt, _predictions)
    r2_metric.set(r2)
    
@api_router.get("/metrics")
async def get_metrics():
    update_metrics()
    return Response(media_type="text/plain",content = prom.generate_latest())

