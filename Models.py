import math
from statsmodels.api import OLS
import numpy as np
from tables import IsDescription

def collinear_adj_prior(exog):
    '''collinearity adjusted dilution prior
    
    Parameters
    ----------
    exog : np.ndarray
        exogenous data, includes a constant

    Returns
    -------
    prob : float
        proportional to prior model probability

    Notes
    -----
    See George (2010); Dilution Priors: Compensating for Model
    Space Redundancy
    
    pi = 0.5
    
    Issues
    ------
    If data does not have rank of at least 3, will not
    calculate the dilution (correlation)
    '''
    
    if exog.shape[1] < 3:
        return 1.
    
    X = exog[:, 1:] #non-constant columns
    
    corr = np.corrcoef(X, rowvar=0)
    
    return max(0,np.linalg.det(corr))

def linear(data, **kwargs):
    '''linear regression model fitted with ordinary least squares
    
    Parameters
    ----------
    data : array or dataframe
        first column is endogenous, second column is
        a column of ones, the rest are exogenous data

    ** Keyword Arguments **

    prior_type : str
        'uniform' or 'collinear adjusted dilution'
    
    Returns
    -------
    rslts : array
        1-d array of parameter coefficients
    '''
    
    prior_type = kwargs.get('prior_type', 'uniform')

    endog = data[:, [0]]
    exog = data[:, 1:]

    model = OLS(endog=endog, exog=exog, missing='drop')
    
    adj = (np.cov(np.hstack((model.wexog, endog)), rowvar=0)[:-1, -1]/ \
            np.var(endog)).reshape((-1, 1))
    
    fit = model.fit()
    
    par_rsquared = fit.params.reshape((-1,1))*adj    
    
    if prior_type == 'uniform':
        prior = 1.
    elif prior_type == 'collinear adjusted dilution':
        prior = collinear_adj_prior(exog)
    else:
        raise ValueError('prior {} not supported'.format(prior_type))
    
    posterior = math.exp(fit.llf)*prior
        
    return np.hstack((fit.nobs, posterior, fit.rsquared, fit.params, fit.pvalues, fit.bse, par_rsquared.flat))