from typing import Any, List, Optional
import datetime

from pydantic import BaseModel
from patient_survival_prediction_model.processing.validation import DataInputSchema

class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[str]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "age": 75,
                        "anaemia": 0,
                        "creatinine_phosphokinase": 0,
                        "diabetes": 0,
                        "ejection_fraction": 0,
                        "high_blood_pressure": 0,
                        "platelets": 0,
                        "serum_creatinine": 0,
                        "serum_sodium": 0,
                        "sex": 0,
                        "smoking": 0,
                        "time": 0 
                    }
                ]
            }
        }
