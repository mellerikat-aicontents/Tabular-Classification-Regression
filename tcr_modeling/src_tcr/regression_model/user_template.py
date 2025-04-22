import numpy as np
import pandas as pd
import math
import sys

from model_manager import Wrapper
# import python package needed to 

# DEFAULT 값 self.config에 맞게 변경하여 담을 것
# rf CLassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
DEFAULT_PARAM = {
    'arg1': 6,
    'arg2': [300, 500],
    'arg3': 1234,
    'tcr_param_mix': 'one_to_one', # one-to-one or all allowed
}

class Your_model:
    def __init__(self, arg1, arg2, arg3, model):
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3
        self.model = model

class TCR_model(Wrapper):
    def __init__(self, model_name, model_type, param_dict):
        model = Your_model(**param_dict)
        super().__init__(model_name, model_type, model)
