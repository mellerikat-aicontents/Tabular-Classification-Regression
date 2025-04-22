from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from .wrapper import Wrapper



class TCR_Sampler(Wrapper):
    def __init__(self, sampler_name, param_dict = {}, config=None):
        if config['sampling']['task_type'] == 'over_sampling':
            sampler_model = RandomOverSampler(**param_dict)
        else: 
            sampler_model = RandomUnderSampler(**param_dict) 
        super().__init__(sampler_name, sampler_model, config)
