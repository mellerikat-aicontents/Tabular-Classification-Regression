from sklearn.ensemble import RandomForestClassifier

from .classifier import Classfier

# DEFAULT 값 self.config에 맞게 변경하여 담을 것
# rf CLassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
DEFAULT_PARAM = {
    'max_depth': 6,
    'n_estimators': [300, 500],
    'random_state': 1234,
    'n_jobs': 1,
    'tcr_param_mix': 'one_to_one',
}

class TCR_model(Classfier):
    def __init__(self, model_name, model_type, param_dict):
        model = RandomForestClassifier(**param_dict)
        super().__init__(model_name, model_type, model)
