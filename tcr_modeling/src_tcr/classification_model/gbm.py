from sklearn.ensemble import GradientBoostingClassifier

from .classifier import Classfier

# DEFAULT 값 self.config에 맞게 변경하여 담을 것
# gbm CLassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
DEFAULT_PARAM = {
    'max_depth': [5, 6, 7],
    'n_estimators': [300, 400, 500],
    'random_state': 1234,
    'tcr_param_mix': 'one_to_one',
}

class TCR_model(Classfier):
    def __init__(self, model_name, model_type, param_dict):
        model= GradientBoostingClassifier(**param_dict)
        super().__init__(model_name, model_type, model)