import warnings

import numpy as np
import statsmodels.api as sm
from scipy.misc import comb

class Context(object):
    '''Functionality for objects that put themselves in a 
    context using the `with` statement.
    
    Notes
    -----
    Taken from PyMC Library
    '''
    def __enter__(self):
        type(self).get_contexts().append(self)
        return self

    def __exit__(self, typ, value, traceback):
        type(self).get_contexts().pop()

    @classmethod
    def get_contexts(cls):
        if not hasattr(cls, "contexts"):
            cls.contexts = []

        return cls.contexts

    @classmethod
    def get_context(cls):
        """Return the deepest context on the stack."""
        try:
            return cls.get_contexts()[-1]
        except IndexError:
            raise TypeError("No context on context stack")  

def modelcontext():
    """return the given model space"""
    
    return ModelSpace.get_context()
  
class ModelSpace(Context):
    '''Builds and stores information about a space of linear models

    Parameters
    ----------
    endog : np.ndarray
        endogenous data
    exog : np.ndarray
        exogenous, or explanatory data. Should not contain a constant
    
    ** Attributes **
 
    endog : np.ndarray
        data to predict using exog and model
    exog : np.ndarray
        data used to predict endog
    K : int
        number of possible regressors
    model : model class
        a model class to estimate
    model_kwargs : keyword arguments
        keyword arguments to pass onto the model class
    fit_kwargs : keyword arguments
        keyword arguments to pass onto the model class's fit method
    attributes : dict
        dictionary of attributes to store from the model's result
        class, which is grabbed when fitting the model
    K : int
        total number of regressors to choose from
    max_models : int
        total number of possible models given number of allowed regressors
        and K
    regressors : array-like
        allowable number of regressors
    self.prior_type: str
        type of model prior
    '''

    def __init__(self, endog, exog, model=sm.OLS):
        self._handle_data(endog, exog)
        self._handle_model(model)
        self.model_kwargs = {}
        self.fit_kwargs = {}
        self.attributes = {'posterior':0, 'prior':0, 'par_rsquared':1, 'bse':1, 'bic':0, 
                       'aic':0, 'rsquared':0, 'rsquared_adj':0, 'params':1, 'visits':0,
                       'nobs':0}
        self.K = exog.shape[1]
        self.max_models = 2**(exog.shape[1])
        self.regressors = np.arange(0,exog.shape[1])
        self.prior_type = 'uniform'
    
    def set_prior_type(self, prior_type):
        '''set prior for model
        
        Parameters
        ----------
        prior_type : str
            'uniform' or 'collinear adjusted dilution'
        '''
        self.prior_type = prior_type
        
    def set_model_kwargs(self, **kwargs):
        '''sets the model's keyword args
        
        Parameters
        ----------
        kwargs : keyword arguments
            keyword arguments to use when instantiating model
        '''
        self.model_kwargs = kwargs

    def set_fit_kwargs(self, kwargs):
        '''sets the model fit method's keyword args
        
        Parameters
        ----------
        kwargs : keyword arguments
            keyword arguments to use when fitting the model
        '''
        self.fit_kwargs = kwargs

    def set_attributes(self, attributes):
        '''sets the attributes to retrieve from model fit

        Parameters
        ----------
        attributes : dict
            dictionary, keys are attribute names, values are 1 if there attribute is 
            related to the regression coefficients
        '''
        self.attributes = attributes

    def set_regressors(self, min_regressors, max_regressors):
        '''sets the minimum and maximum number of regressors to consider

        Parameters
        ----------
        min_regressors : int
            minimum number of regressors for sampled models
        max_regressors : int
            maximum number of regressors for sampled models
        '''
        if (not isinstance(min_regressors, int)) | (not isinstance(max_regressors, int)):
            raise ValueError('min and max number of regressors must be integers')
        if min_regressors > max_regressors:
            raise ValueError('min number of regressors greater than max number')
        self.regressors = np.arange(min_regressors, max_regressors+1)
        sizes = [comb(self.K, k) for k in self.regressors]
        self.max_models = int(sum(sizes))

    def _handle_model(self, model):
        '''handles the model

        Parameters
        ----------
        model : a model class
            a model class, must take endog and exog as parameters
        '''
        self.model = model
    
    def _handle_data(self, endog, exog):
        '''handles the data
        
        Parameters
        ----------
        endog : np.ndarray
            2-d array of shape (-1,1)
        exog : np.ndarray
            2-d array of shape (-1,K), does not contain constants
        '''
        
        if not min(exog.std(axis=0)) > 0:
            raise ValueError('Data contains column with too little variation') 
        if endog.shape[0] != exog.shape[0]:
            raise ValueError('exog and endog data not of same length')
        
        self.exog = sm.add_constant(exog, prepend=True)
        self.endog = endog
        self.K = exog.shape[1]

    def fit(self, regressors):
        '''fits the model on the specified regressors

        Parameters
        ----------
        regressors : array-like
            list or array of regressors (columns in exog)
        kwargs : keyword arguments
            arguments to pass onto the model's fit method
        '''
        
        model = self.model(endog=self.endog, exog=self.exog[:,regressors], \
                            missing='drop', **self.model_kwargs)
        
        # Quality
        if len(model.endog) - model.exog.shape[1] < 5:
            warnings.warn('Model has fewer than 5 degrees of freedom')
        
        rslts = model.fit_for_ma(self.prior_type, **self.fit_kwargs)
        
        # Assigning assumed values, based on model selected by regressors
        if self.model == sm.MNLogit:
            for attr in self.attributes:
                if self.attributes[attr]:
                    values = getattr(rslts, attr)
                    values_ = np.zeros((self.exog.shape[1],2))
                    values_[regressors] = values
                    values = values_
                    setattr(rslts, attr, values)
        else:
            for attr in self.attributes:
                if self.attributes[attr]:
                    values = getattr(rslts, attr)
                    values_ = np.zeros(self.exog.shape[1])
                    values_[regressors] = values
                    values = values_
                    setattr(rslts, attr, values)
                
        return rslts