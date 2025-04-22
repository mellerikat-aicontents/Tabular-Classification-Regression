import sys
import json
import numpy as np
import pandas as pd

import sys
from pathlib import Path
CURRENT_PATH = str(Path(__file__).parent)
if CURRENT_PATH not in sys.path:
    sys.path.append(CURRENT_PATH)

import sampling.sampler.nearmiss as nearmiss
import sampling.sampler.random as random_sampler
import sampling.sampler.smote as smote



class sampler_manager:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        self.sampler_modules = {}
        self.sampler_modules['under'] = {}
        self.sampler_modules['over'] = {}
        self.load_sampler(config)
        
    def load_sampler(self, config):
        self.sampler_modules['under']['nearmiss'] = nearmiss.TCR_Sampler('nearmiss', config=config)
        self.sampler_modules['under']['random'] = random_sampler.TCR_Sampler('random', config=config)

        self.sampler_modules['over']['random'] = random_sampler.TCR_Sampler('random', config=config)
        self.sampler_modules['over']['smote'] = smote.TCR_Sampler('smote', config=config)
        
        
    def set_param(self, sampling_type, sampler_name, param_dict):
        self.save_info(f'sampling model: {self.sampler_modules[sampling_type][sampler_name].sampler_model}')
        self.sampler_modules[sampling_type][sampler_name].set_param(param_dict)

    
    def under_fit_resampling(self,  sampler_name, x_data, y_data):
        sampling_type = 'under'
        x_res, y_res = self.sampler_modules[sampling_type][sampler_name].fit_resample(x_data, y_data)
        changed_idx = self.sampler_modules[sampling_type][sampler_name].get_sample_indices()
        return x_res, y_res, changed_idx
        
    def over_fit_resampling(self, sampler_name, x_data, y_data):
        sampling_type = 'over'
        x_res, y_res = self.sampler_modules[sampling_type][sampler_name].fit_resample(x_data, y_data)
        
        origin_idx = len(x_data)
        new_x = x_res[origin_idx:]
        new_y = y_res[origin_idx:]
        
        over_df = pd.concat([new_x, new_y], axis=1)
        return over_df         
        
    def save_info(self, msg):
        self.logger['info'](msg)
     
    def save_warning(self, msg):
        self.logger['warning'](msg)
 
    def save_error(self, msg):
        self.logger['error'](msg)
