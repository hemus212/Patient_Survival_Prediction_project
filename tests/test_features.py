
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from patient_survival_prediction_model.config.core import config
from patient_survival_prediction_model.processing.features import  Mapper, OutlierHandler


def test_creatinine_phosphokinase_variable_outlierhandler(sample_input_data):
    # Given
    encoder = OutlierHandler(variable = config.self_model_config.creatinine_phosphokinase_var)
    q1, q3 = np.percentile(sample_input_data[0]['creatinine_phosphokinase'], q=[25, 75])
    iqr = q3 - q1
    assert sample_input_data[0].loc[5813, 'creatinine_phosphokinase'] > q3 + (1.5 * iqr)

    # When
    subject = encoder.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[5813, 'creatinine_phosphokinase'] <= q3 + (1.5 * iqr)

