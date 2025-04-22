class Wrapper:
    def __init__(self, sampler_name, sampler_model, config):
        self.sampler_name = sampler_name
        self.sampler_model = sampler_model
        self.config = config

    def fit_resample(self, x_data, y_data):
        x_res, y_res = self.sampler_model.fit_resample(x_data, y_data)
        return x_res, y_res


    def get_sample_indices(self):
        return self.sampler_model.sample_indices_
        

    def set_param(self, sampler_param):
        print('set param')
        print('before', self.sampler_model.get_params())
        self.sampler_model = self.sampler_model.set_params(**sampler_param)
        print('after', self.sampler_model.get_params())

