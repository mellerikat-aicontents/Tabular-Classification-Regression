import numpy as np
from ngboost import NGBClassifier
from ngboost.distns.categorical import k_categorical

from .classifier import Classfier

# DEFAULT 값 self.config에 맞게 변경하여 담을 것
# ngb CLassifier
#    Parameters:
#        Dist              : assumed distributional form of Y|X=x.
#                            A distribution from ngboost.distns, e.g. Bernoulli
#        Score             : rule to compare probabilistic predictions P̂ to the observed data y.
#                            A score from ngboost.scores, e.g. LogScore
#        Base              : base learner to use in the boosting algorithm.
#                            Any instantiated sklearn regressor, e.g. DecisionTreeRegressor()
#        natural_gradient  : logical flag indicating whether the natural gradient should be used
#        n_estimators      : the number of boosting iterations to fit
#        learning_rate     : the learning rate
#        minibatch_frac    : the percent subsample of rows to use in each boosting iteration
#        col_sample        : the percent subsample of columns to use in each boosting iteration
#        verbose           : flag indicating whether output should be printed during fitting
#        verbose_eval      : increment (in boosting iterations) at which output should be printed
#        tol               : numerical tolerance to be used in optimization
#        random_state      : seed for reproducibility. See
#                            https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn

DEFAULT_PARAM = {
    'col_sample': [0.6, 0.7, 0.8],
    'n_estimators': [100, 200, 300],
    'natural_gradient': False,
    'random_state': 1234,
    'verbose': False,
    'tcr_param_mix': 'one_to_one',
}

class TCR_model(Classfier):
    def __init__(self, model_name, model_type, param_dict):
        model = None
        self.param_dict = param_dict
        super().__init__(model_name, model_type, model)

    def fit(self, train_data, label):
        '''gi
            Model 학습 (필요할 경우 override)
        '''
        self.model = NGBClassifier(Dist=k_categorical(label.nunique()), **self.param_dict)
        self.model.fit(train_data, label)
        setattr(self.model, 'classes_', np.arange(label.nunique()))
        self.classes = self.model.classes_
        self.trained = True
