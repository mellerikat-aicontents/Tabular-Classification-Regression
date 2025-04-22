from ngboost import NGBRegressor

from .regressor import Regressor

# DEFAULT 값 self.config에 맞게 변경하여 담을 것
# ngb Regressor
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
    'n_estimators': [100, 300],
    'col_sample': [0.6, 0.8],
    'natural_gradient': False,
    'verbose': False,
    'random_state': 1234,
    'tcr_param_mix': 'one_to_one',
}

class TCR_model(Regressor):
    def __init__(self, model_name, model_type, param_dict):
        model = NGBRegressor(**param_dict)
        super().__init__(model_name, model_type, model)
