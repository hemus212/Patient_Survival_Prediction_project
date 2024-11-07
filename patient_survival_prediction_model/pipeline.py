import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline

from patient_survival_prediction_model.config.core import config
from patient_survival_prediction_model.processing.features import Mapper
from patient_survival_prediction_model.processing.features import OutlierHandler
from xgboost import XGBClassifier

patient_survival_prediction_pipe = Pipeline([

    ######## Handle outliers ########
    ('handle_outliers_creatinine_phosphokinase', OutlierHandler(variable = config.self_model_config.creatinine_phosphokinase_var)),
    ('handle_outliers_ejection_fraction', OutlierHandler(variable = config.self_model_config.ejection_fraction_var)),
    ('handle_outliers_platelets', OutlierHandler(variable = config.self_model_config.platelets_var)),
    ('handle_outliers_serum_creatinine', OutlierHandler(variable = config.self_model_config.serum_creatinine_var)),
    ('handle_outliers_serum_sodium', OutlierHandler(variable = config.self_model_config.serum_sodium_var)),
    
    # xgb_clf
    ('model_xgb_clf', XGBClassifier(n_estimators = config.self_model_config.n_estimators, 
                                       max_depth = config.self_model_config.max_depth,
                                       max_leaves = config.self_model_config.max_leaves,
                                      random_state = config.self_model_config.random_state))
    
    ])
