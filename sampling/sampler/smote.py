from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTEN

from .wrapper import Wrapper

class TCR_Sampler(Wrapper):
    def __init__(self, sampler_name, param_dict = {}, config=None):
        self.categorical_columns = config['readiness']['categorical_columns']
        self.numeric_columns     = config['readiness']['numeric_columns']
        if len(self.categorical_columns) > 0 and len(self.numeric_columns) > 0:
            # SMOTENC 초기화를 위해 필수 parameter인 categorical_features를 추가함
            self.sampler_model        = SMOTENC({'categorical_features':[]}) 
            self.categorical_features = [config['readiness']['x_columns'].index(cat) for cat in self.categorical_columns]
        elif len(self.categorical_columns) > 0 and len(self.numeric_columns) == 0:
            self.sampler_model = SMOTEN(**param_dict)
        elif len(self.categorical_columns) == 0 and len(self.numeric_columns) > 0:
            self.sampler_model = SMOTE(**param_dict)
        super().__init__(sampler_name, self.sampler_model, config)


    def set_param(self, sampler_param):
        print('set param')
        print('before', self.sampler_model.get_params())
        if len(self.categorical_columns) > 0 and len(self.numeric_columns) > 0:
            sampler_param['categorical_features'] = self.categorical_features 
        self.sampler_model = self.sampler_model.set_params(**sampler_param)
        print('after', self.sampler_model.get_params())
