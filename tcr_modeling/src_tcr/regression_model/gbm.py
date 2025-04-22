from sklearn.ensemble import GradientBoostingRegressor

from .regressor import Regressor

# DEFAULT 값 self.config에 맞게 변경하여 담을 것
# gbm Regressor: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

DEFAULT_PARAM = {
    'max_depth': [5, 7],
    'n_estimators': [300, 500],
    'random_state': 1234,
    'tcr_param_mix': 'one_to_one',
}

class TCR_model(Regressor):
    def __init__(self, model_name, model_type, param_dict):
        model = GradientBoostingRegressor(**param_dict)
        super().__init__(model_name, model_type, model)
