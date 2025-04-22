from catboost import CatBoostClassifier

from .classifier import Classfier

# DEFAULT 값 self.config에 맞게 변경하여 담을 것
# cb CLassifier: https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier#description1
DEFAULT_PARAM = {
    'max_depth': [5, 7, 9],
    'n_estimators': [100, 300, 500],
    'thread_count': 6,
    'allow_writing_files': False,
    'verbose': 0,
    'random_state': 1234,
    'tcr_param_mix': 'one_to_one',
}

class TCR_model(Classfier):
    def __init__(self, model_name, model_type, param_dict):
        model = CatBoostClassifier(**param_dict)
        super().__init__(model_name, model_type, model)

    def fit(self, train_data, label):
        '''
            Model 학습 (필요할 경우 override)
        '''
        self.model.fit(train_data, y=label, logging_level='Silent')
        self.classes = self.model.classes_
        self.trained = True
