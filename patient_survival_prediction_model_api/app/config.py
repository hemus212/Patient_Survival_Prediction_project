import sys
from typing import List
import pandas as pd
from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings
from patient_survival_prediction_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config
from pathlib import Path
import prometheus_client as prom

test_data = pd.read_csv(Path(f"{DATASET_DIR}/heart_failure_clinical_records_dataset.csv")) 
r2_metric = prom.Gauge('patient_survival_score', 'R2 score for random 100 testsamples')
class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"

    # Meta
 
    # BACKEND_CORS_ORIGINS is a comma-separated list of origins
    # e.g: http://localhost,http://localhost:4200,http://localhost:3000
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",  # type: ignore
        "http://localhost:8000",  # type: ignore
        "https://localhost:3000",  # type: ignore
        "https://localhost:8000",  # type: ignore
    ]

    PROJECT_NAME: str = "Patient Survival Prediction API"

    class Config:
        case_sensitive = True

settings = Settings()
