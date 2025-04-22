from imblearn.under_sampling import NearMiss
from .wrapper import Wrapper


class TCR_Sampler(Wrapper):
    def __init__(self, sampler_name, param_dict = {}, config=None):
        sampler_model = NearMiss(**param_dict)
        super().__init__(sampler_name, sampler_model, config)
