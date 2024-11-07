"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from patient_survival_prediction_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # When
    result = make_prediction(input_data=sample_input_data)
    
    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, str), "Predictions should be a string message"
    assert "The patient is predicted" in predictions, "Prediction message should indicate patient status"
    assert result.get("errors") is None, "There should be no errors in the result"
    #assert len([predictions]) == expected_num_of_predictions, f"Expected {expected_num_of_predictions} predictions, got {len([predictions])}"
    assert result.get("version") is not None, "Version should be present in the result"

    _predictions = list(predictions)
    y_true = sample_input_data[1]

    r2 = r2_score(y_true, _predictions)
    mse = mean_squared_error(y_true, _predictions)

    assert r2 > 0.8
    assert mse < 3000.0
