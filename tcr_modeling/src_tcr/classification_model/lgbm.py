from lightgbm import LGBMClassifier

from .classifier import Classfier

# DEFAULT 값 self.config에 맞게 변경하여 담을 것
# lgb CLassifier: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html

DEFAULT_PARAM = {
    'max_depth': [5, 7, 9],
    'n_estimators': [300, 400, 500],
    'random_state': 1234,
    'verbose': -1,
    'n_jobs': 1,
    'tcr_param_mix': 'one_to_one',
}

class TCR_model(Classfier):
    def __init__(self, model_name, model_type, param_dict):
        model = LGBMClassifier(**param_dict)
        super().__init__(model_name, model_type, model)

    def fit(self, train_data, label):
        '''
            Model 학습 (필요할 경우 override)
        '''

        self.model.fit(train_data, label)
        self.classes = self.model.classes_
        self.trained = True
