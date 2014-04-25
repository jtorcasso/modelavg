import numpy as np

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
        return 1
    
    X = exog[:, 1:] #non-constant columns
    
    corr = np.corrcoef(X, rowvar=0)
    
    return max(0,np.linalg.det(corr))