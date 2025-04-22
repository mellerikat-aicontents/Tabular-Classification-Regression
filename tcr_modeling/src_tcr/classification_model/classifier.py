import sys

import joblib
import numpy as np

from model_manager import Wrapper

class Classfier(Wrapper):
    def __init__(self, model_name, model_type, model):
        super().__init__(model_name, model_type, model)
        self.classes = None

    def fit(self, train_data, label):
        '''
            Model 학습 (필요할 경우 override)
        '''
        self.model.fit(train_data, label)
        self.classes = self.model.classes_
        self.trained = True

    def load(self, path):
        '''
            Model 로드 (필요할 경우 override)
        '''
        self.model = joblib.load(path)
        self.classes = np.array(self.model.classes_)
        self.trained = True

    def predict(self, inference_data):
        '''
            Model 예측 (필요할 경우 override)
        '''
        y_prob = self.model.predict_proba(inference_data)
        y_pred = self.classes[y_prob.argmax(axis=1)]
        y_pred = np.column_stack((y_prob, y_pred))

        return y_pred

    def get_col_names(self, y_cols):
        # One-hot Vetor인 Case (추후 고려? or 지원 X)
        if isinstance(y_cols, list):
            col_names = [f'TCR-prob_{target_class}' for target_class in self.classes]
        else:
            col_names = [f'TCR-prob_{target_class}' for target_class in self.classes] + [f'TCR-pred_{y_cols}']

        return col_names
