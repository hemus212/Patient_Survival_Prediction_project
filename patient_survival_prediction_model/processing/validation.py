import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

from datetime import datetime
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from patient_survival_prediction_model.config.core import config


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    validated_data = input_df[config.self_model_config.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs = validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors

class DataInputSchema(BaseModel):
    age: Optional[int]
    high_blood_pressure: Optional[int]
    anaemia: Optional[int]
    creatinine_phosphokinase: Optional[int]
    diabetes: Optional[int]
    ejection_fraction: Optional[int]
    platelets: Optional[int]
    sex: Optional[int]
    serum_creatinine: Optional[int]
    serum_sodium: Optional[int]
    smoking: Optional[int]
    time: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]