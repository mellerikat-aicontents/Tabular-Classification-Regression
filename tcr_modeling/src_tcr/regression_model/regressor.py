import sys

from model_manager import Wrapper

class Regressor(Wrapper):
    def predict(self, inference_data):
        '''
            Model 예측 (필요할 경우 override)
        '''
        y_pred = self.model.predict(inference_data)
        return y_pred
