import math
import statsmodels.api as sm
import numpy as np
from ModelPriors import collinear_adj_prior

def fit_for_ma(self, prior_type=None, **kwargs):
    '''an alternative fit method for statsmodels models
    
    Parameters
    ----------
    self : statsmodels model class
        class
    
    Returns
    -------
    rslts : sm.regression.RegressionResults instance
        regression results instance with additional results
    '''
    
    endog = self.endog.reshape((-1,1))
    
    if hasattr(self, 'wexog'):
        adj = (np.cov(np.hstack((self.wexog,endog)),rowvar=0)[:-1,-1]/\
            np.var(endog)).reshape((-1,1))
    else:
        adj = None
    
    rslts = self.fit(**kwargs)
    
    if adj is not None:
        rslts.par_rsquared = rslts.params.reshape((-1,1))*adj    
    
    if hasattr(rslts, 'llf') & (prior_type is not None):
        if prior_type == 'uniform':
            rslts.prior = 1
        elif prior_type == 'collinear adjusted dilution':
            rslts.prior = collinear_adj_prior(self.exog)    
        rslts.posterior = math.exp(rslts.llf)*rslts.prior

    rslts.visits = 0
        
    return rslts

# Overwriting existing fit methods
for class_ in [sm.OLS, sm.GLS, sm.WLS, sm.Probit, sm.Logit, sm.MNLogit]:
    class_.fit_for_ma = fit_for_ma