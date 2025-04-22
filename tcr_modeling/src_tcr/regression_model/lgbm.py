from lightgbm import LGBMRegressor

from .regressor import Regressor

# DEFAULT 값 self.config에 맞게 변경하여 담을 것
# lgbm Regressor: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html

DEFAULT_PARAM = {
    'max_depth': [5, 7],
    'n_estimators': [300, 500],
    'random_state': 1234,
    'verbose': -1,
    'n_jobs': 1,
    'tcr_param_mix': 'one_to_one',
}

class TCR_model(Regressor):
    def __init__(self, model_name, model_type, param_dict):
        model = LGBMRegressor(**param_dict)
        super().__init__(model_name, model_type, model)
