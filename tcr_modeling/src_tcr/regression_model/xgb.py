from xgboost import XGBRegressor

from .regressor import Regressor

# DEFAULT 값 self.config에 맞게 변경하여 담을 것
# cb CLassifier: https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier#description1
DEFAULT_PARAM = {
    'max_depth': [5, 6, 7],
    'n_estimators': [100, 300, 500],
    'n_jobs': 1,
    'verbosity': 0,
    'random_state': 1234,
    'tcr_param_mix': 'one_to_one',
}

class TCR_model(Regressor):
    def __init__(self, model_name, model_type, param_dict):
        model = XGBRegressor(**param_dict)
        super().__init__(model_name, model_type, model)