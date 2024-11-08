import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np
import logging

from patient_survival_prediction_model import __version__ as _version
from patient_survival_prediction_model.config.core import config
from patient_survival_prediction_model.processing.data_manager import load_pipeline
from patient_survival_prediction_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
patient_survival_prediction_pipe = load_pipeline(file_name = pipeline_file_name)

def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    logging.basicConfig(filename="app.log", level=logging.DEBUG)
    try:
        validation_result = validate_inputs(input_df = pd.DataFrame(input_data))
        if isinstance(validation_result, tuple) and len(validation_result) == 2:
            validated_data, errors = validation_result
        else:
            raise ValueError("validate_inputs did not return the expected tuple")
        
        validated_data = validated_data.reindex(columns = config.self_model_config.features)
        
        results = {"predictions": None, "version": _version, "errors": errors}
        
        if not errors:
            predictions = patient_survival_prediction_pipe.predict(validated_data)
            predictions = np.where(predictions == 0, 
                                   "The patient is predicted to not have a death event.",
                                   "The patient is predicted to have a death event.")
            results = {"predictions": predictions, "version": _version, "errors": errors}
        return results
    
    except Exception as e:
        logging.error("Error creating prediction", exc_info=True)
        print(f"Error: {str(e)}")
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":

    data_in = {'age': [30],'anaemia': [0], 'creatinine_phosphokinase': [6200], 'diabetes': [1],
               'ejection_fraction': [58], 'high_blood_pressure': [0],'platelets': [60000.0], 'serum_creatinine': [0.9], 'serum_sodium': [153], 'sex': [1],'smoking': [0], 'time': [100]}
    make_prediction(input_data = data_in)