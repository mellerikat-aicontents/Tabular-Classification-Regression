from catboost import CatBoostRegressor

from .regressor import Regressor

# DEFAULT 값 self.config에 맞게 변경하여 담을 것
# cb Regressor: https://catboost.ai/en/docs/concepts/python-reference_catboostregressor

DEFAULT_PARAM = {
    'max_depth': [5, 7, 9],
    'n_estimators': [100, 300, 500],
    'thread_count': 6,
    'allow_writing_files': False,
    'verbose': 0,
    'random_state': 1234,
    'tcr_param_mix': 'one_to_one',
}

class TCR_model(Regressor):
    def __init__(self, model_name, model_type, param_dict):
        model = CatBoostRegressor(**param_dict)
        super().__init__(model_name, model_type, model)
