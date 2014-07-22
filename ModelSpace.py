import warnings

from pandas import DataFrame, Series
import numpy as np
from Models import linear
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
    data : array-like
        data to be used in the model, the first column is the 
        endogenous variable, the rest are exogenous
    model : function
        function taking data and keyword arguments (kwargs)
        and returns an array of coefficient estimates
    kwargs : keyword arguments
        keyword arguments to pass onto the model
    keep : list
        list of columns in data to always keep in the model, specified
        by horizontal position, starting at 0

    
    ** Attributes **
 
    maxm : int
        total number of possible models given number of allowed regressors
        and K
    k : array-like
        allowable number of regressors
    '''

    def __init__(self, data, k=None, keep=[0, 1], model=linear, kwargs={}):

        if not set([0,1]).issubset(set(keep)):
            raise ValueError('keep should include [0, 1]')
        if k is not None:
            assert min(k) >= len(keep)
            
        self.data = self._handle_data(data)
        self.keep = tuple(keep)
        self.choices = np.array([i for i in xrange(data.shape[1]) if i not in keep])
        self.model = model
        self.kwargs = kwargs
        self._build(k)

    @staticmethod
    def _handle_data(data):
        '''handles the data'''

        if not isinstance(data, np.ndarray):
            raise ValueError('Data should be NumPy Array')

        if not len(data.shape) == 2:
            raise ValueError('Data should be two dimensional')

        if data[:,1].std() != 0:
            raise ValueError('column at index 1 should be constant')

        return data

    def _build(self, k):
        '''constructs model space parameters base on given attributes'''

        self.k = np.arange(1, len(self.choices)+1, 1) if k is None else np.array(k)
        self.maxm = int(sum([comb(len(self.choices), i-len(self.keep)+2) for i in self.k]))
    
    def fit(self, columns):
        '''fits the model with the specified columns

        Parameters
        ----------
        columns : array-like
            list or array of column numbers in data
        '''

        return self.model(data=self.data[:, columns], **self.kwargs)