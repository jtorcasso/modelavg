'''Toolkit for Model Averaging



Author:       Jake Torcasso
License:      BSD (3-clause)


Issues
------
1. How reasonable is calculating the likelihood to estimate
the Bayesian Information Criterion (BIC) for small samples?

2. Make sure errors are caught properly when making results: 
Does the model fit supply the necessary ingredients for calculating
model priors, posteriors and partial r-squareds?

3. Missing values, right now just drops them on each run of a model.
What happens if a model encounters an error due to missing values?

Notes
-----
1. Does not calculate true probabilities. If
run on a subspace of the model space, calculates
probabilities relative to this space. 

2. The BIC approximation to 2log(Pr(Data|M)) is only
valid for large samples. AND we need to characterize
a likelihood and therefore require parametric assumptions.

General References : 

A. E. Raftery, D. Madigan, and J. A. Hoeting. "Bayesian
Model Averaging for Linear Regression Models". 1997

J. A. Hoeting, D. Madigan, A. E. Raftery, and C. T. Volinsky.
"Bayesian Model Averaging: A Tutorial". 1999
'''

# Future
from __future__ import print_function, division

# Standard Library
import random
import multiprocessing as mp

# Third Party
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pylab as plt

from ModelSpace import *

def SampleAll():
    '''samples from all of the models'''

    model_space = modelcontext()
    sample = {}
    regressors = model_space.regressors
    for k in range(len(regressors)):
        for cols in itertools.combinations(regressors, k+1):
            sample.update({str(cols):model_space.fit(draw)})

    return sample

def random_draw(num_regressors, K):
    '''draws a set of regressors at random
    
    Parameters
    ----------
    num_regressors : array-like
        number of regressors to go in model
    K : int
        total number of regressors to select from
    
    Returns
    -------
    draw : list
        set of regressors
    '''
    k = np.random.choice(num_regressors)
    
    cols = sorted(np.random.choice(xrange(1, K+1), size=k, replace=False))
    
    return [0] + cols

def RandomSample(draws, seed=1234):
    '''draws a random sample from ModelSpace in current context
    
    Parameters
    ----------
    draws : int
        number of draws
    seed : int
        seed number for random numbers
    
    '''
    model_space = modelcontext()
    np.random.seed(seed)
    sample = {}
    while len(sample) < draws:
        draw = random_draw(model_space.regressors, model_space.K)
        sample.update({str(draw):model_space.fit(draw)})
    
    return sample
        
def pRandomSample(draws, seed=1234, threads=1):
    '''parallel random sampler
    
    Parameters
    ----------
    draws : int
        number of draws
    seed : int
        seed number for random numbers
    threads : int
        number of threads to spawn for sampling
    
    Notes
    -----
    will create random sample of models for `draws` number
    of draws on each thread
    
    '''    
    
    argset = zip([draws]*threads, [seed+i for i in range(threads)])
    
    p = mp.Pool(threads)
    
    jobs = [p.apply_async(RandomSample, args) for args in argset]
    samples = [j.get() for j in jobs]
    
    sample = {}
    for s in samples:
        sample.update(s)
    
    p.close()
    
    return sample

def mcmc_draw(last_draw, model_space, cache={}):
    '''moves to next model in markov chain sampler for model space
    
    Parameters
    ----------
    last_draw : list
        set of regressors from previous draw
    model_space : model space instance
        the model space
    cache : dict
        dictionary to store regression results
    
    Returns
    -------
    draw : list
        set of regressors
    '''
    K = model_space.K
    num_regressors = model_space.regressors    
    
    if last_draw is None:
        regressors = random_draw(num_regressors, K)
        rslts = model_space.fit(regressors)
        return regressors, rslts
    
    width = K + 1
    prev = np.zeros(width)
    prev[last_draw] = 1
    prev = prev.reshape((-1,1))
    
    neighbors = abs(np.diag(np.ones(width)) - prev)[:,1:]
    
    neighbors = neighbors[:,np.any([neighbors.sum(axis=0) == i \
                    for i in num_regressors], axis=0)]
    
    neighbors = pd.DataFrame(neighbors)
    
    draw = random.choice(xrange(neighbors.shape[1]))
    
    proposal = list(neighbors[draw][neighbors[draw]==1].index)
    
    
    if str(proposal) in cache:
        rslts = cache[str(proposal)]
    else:
        rslts = model_space.fit(proposal)
    
    prob = min(1, rslts.posterior/cache[str(last_draw)].posterior)
    if np.random.choice([True, False], p=[prob, 1 - prob]):
        return proposal, rslts
    else:
        rslts = cache[str(last_draw)]
        return last_draw, rslts

def MCMC(visits, burn=0, thin=1, seed=1234):
    '''markov chain monte carlo sampler for model space
    
    Parameters
    ----------
    visits : int
        number of visits in chain
    burn : int
        number of visits to burn from beginning of chain
    thin : int
        related to fraction of visits kept in chain
    seed : int
        seed for random number
    '''
    np.random.seed(seed)        
    model_space = modelcontext()
    
    if burn >= visits:
        raise ValueError('burn must be fewer than total visits')
    if thin < 1:
        raise ValueError('thin must be an integer 1 or greater')

    cache = {}
    last_draw = None
    num_visits = 0
    while num_visits < visits:

        regressors, rslts = mcmc_draw(last_draw, model_space, cache)
        cache.update({str(regressors):rslts})
        last_draw = regressors
        num_visits += 1

        if num_visits <= burn:
            continue
        elif num_visits == burn:
            cache = {}
            continue
        if (num_visits - burn)%thin == 0:
            
            rslts.visits += 1
            
    
    return {key:cache[key] for key in cache if cache[key].visits > 0}
    
def pMCMC(visits, burn=0, thin=1, seed=1234, threads=1):
    '''parallel markov chain monte carlo sampler for model space
    
    Parameters
    ----------
    visits : int
        number of visits in chain
    burn : int
        number of visits to burn from beginning of chain
    thin : int
        related to fraction of visits kept in chain
    seed : int
        seed for random number
    threads : int
        number of threads to spawn for sampling
    
    Notes
    -----
    will run a markov chain with `visits` on every thread
    specified
    
    '''    
    
    argset = zip([visits]*threads, [burn]*threads, [thin]*threads, 
                 [seed+i for i in range(threads)])
    
    p = mp.Pool(threads)
    
    jobs = [p.apply_async(MCMC, args) for args in argset]
    samples = [j.get() for j in jobs]
    
    # aggregating results (summing up visits)
    models = []
    for s in samples:
        models.extend(s.keys())
    sample = {}
    for m in list(set(models)):
        visits = 0
        for s in samples:
            if m in s:
                rslts = s[m]
                visits += rslts.visits
        rslts.visits = visits
        sample.update({m:rslts})
    
    p.close()
    p.join()
    
    return sample
  

class Trace(object):
    '''store and aggregate results from a sample of models
    
    Parameters
    ----------
    results : dictionary
        keys are character strings which are attributes of 
        a model's result class, values are (mostly) DataFrames
        which contain this attribute for all models considered

    ** Attributes **
    
    results : dictionary
        keys are strings, value is a pd.DataFrame
    maxModels : int
        total number of possible models
    size : int
        total number of models present in the trace
    '''
    
    def __init__(self, sample):
        self.ids = sample.keys()
        self.ids.sort()
        self.id_keys = {j:i for i,j in enumerate(self.ids)}
        self.results = self._handle_sample(sample, modelcontext())
        self.size = len(sample)
        self.maxModels = modelcontext().max_models

    def _handle_sample(self, sample, model_space):
        '''organizes results from a sample of models into pandas objects
    
        Parameters
        ----------
        sample : dict
            dictionary from sampling functions
        '''
        
        formatted_results = {}        
        for attr in model_space.attributes:
            formatted_results[attr] = np.array(\
                    [getattr(sample[m],attr) for m in self.ids])
        
        # Normalizing
        for attr in ['posterior', 'visits', 'prior']:
            if attr in formatted_results:
                formatted_results[attr] /= formatted_results[attr].sum()
        
        return formatted_results

    def get_raw_result(self, key):
        '''distribution of results
        
        Parameters
        ----------
        key : str
            result key
        
        Returns
        -------
        params : pd.DataFrame or pd.Series
            distrubitions of parameters across models
        '''
        
        return self.results[key]
    
    def best_model(self, over='posterior'):
        '''finds the set of predictors with which maximizes
        the parameter specified by `over`
        
        Parameters
        ----------
        over : str
            result key, specifying a particular attribute of the
            model, usually some form of model fit
        
        Returns
        -------
        Xcols : tuple
            columns in data
        '''
        
        return self.ids[pd.DataFrame(self.results[over])[0].idxmax()]
    
    def _handle_result(self, key, index=None, column=None):
        '''statistic for a model
        
        Parameters
        ----------
        key : str
            result key
        index : str
            index in result dataframe, identifies a model
            if None, returns across all indices
            ex. '0,1,2,3'
        column : int
            column number in data frame, identifies a parameter
            if None, returns results across all columns
            
        Returns
        -------
        params : array-like
            estimated parameter(s)
        '''
        if (column is None) & (index is None):
            rslts = self.results[key]
        elif column is None:
            rslts = self.results[key][self.id_keys[index]]
        elif index is None:
            rslts = self.results[key][:,column]
        else:
            rslts = self.results[key][:,column][self.id_keys[index]]
        
        if 1 in rslts.shape:
            rslts = rslts.flatten()
            if 1 in rslts.shape:
                rslts = rslts[0]
        return rslts
    
    def average(self, key=None, params=None, weight=None):
        '''computes average over set of models
        
        Parameters
        ----------
        key : str
            result key
        params : pandas object
            must have as index model identifiers, e.g.
            a string '0,1,2'
        weight : str or None
            if str, specifies an attribute to use as a weight
            
        Returns
        -------
        average : array-like or float
            weighted averages of parameters across models
        
        Notes
        -----
        E[B|Data] = B_1*Pr(M_1|Data) + ... + B_K*Pr(M_K|Data)
        '''
        
        if (key is not None) & (params is None):
            params = self._handle_result(key)
        elif (params is not None):
            assert (isinstance(params, np.ndarray))
        else:
            raise ValueError('Must specify key or params')
        
        if weight is not None:
            weight = self._handle_result(weight)
        
        if params.ndim == 1:
            return np.average(params, weights=weight)
        else:
            return np.average(params, weights=weight, axis=0)
        
    def bma_coeff_std(self, weight):
        '''computes standard deviation of BMA coefficients
        
        Returns
        -------
        std : np.ndarray
            ma estimates of standard deviation
        
        Notes
        -----
        Var[B|Data] = (Var[B|Data,M_1] + E[B|Data,M_1]^2)Pr(M_1|Data) + ... +
              (Var[B|Data,M_K] + E[B|Data,M_K]^2)Pr(M_K|Data) -
              E[B|Data]^2
        '''
        
        assert ('params' in self.results)
        assert ('bse' in self.results)
        assert (weight in self.results)        
        
        coeff = self.average('params', weight=weight)
        
        var = np.square(self._handle_result('bse'))
        
        coeff_square = np.square(self._handle_result('params'))
        
        return np.sqrt(self.average(params=(var+coeff_square),weight=weight) - np.square(coeff))
        
    
    def plot(self, key, weight=None, col=None):
        '''plot results across models
        
        Parameters
        ----------
        key : str
            result key
        weight : str
            the weight to use for the plot
        col : int or None
            if None, key must specify a result with a single
            value for each model, if int, 
            specifies a column (coefficient number)
        '''
        
        names = {'rsquared_adj':'Adjusted R-Squared',
                 'rsquared':'R-Squared',
                 'params':'Estimated Regression Coefficient',
                 'par_rsquared':'Partial R-Squared of Predictor'}
        
        if key in ['visits', 'posterior', 'prior']:
            
            self._plot_weight(key)
        
        elif key in ['rsquared_adj', 'rsquared']:
            
            self._plot_hist(key, names[key], weight)
        
        elif key in ['params', 'par_rsquared']:
            
            if not isinstance(col, int):
                raise ValueError("'col' must be an integer")
                
            self._plot_hist(key, names[key], weight, col)
            
    def _plot_weight(self, key, numModels=30):
        '''plots posterior across models
        
        Parameters
        ----------
        numModels : int
            number of models to plot statistics for
        '''
        weight = list(self._handle_result(key))
        weight.sort(reverse=True)
    
        plt.bar(left=range(1, len(weight[:numModels])+1), \
                height=weight[:numModels], color='k')
        
        plt.xlabel('Model by Rank Order', fontsize=15)
        plt.ylabel('Weight ({})'.format(key), fontsize=15)
        
        plt.tick_params(axis='both', labelsize=15, \
                    top='off', right='off')
                    
        plt.text(x=plt.xlim()[1]*1./2, y=plt.ylim()[1]*8.5/10, \
                s=('{0}{1}\n{2}{3}\n{4}{5}'.format(\
                'Models Shown:       ', numModels, \
                'Models Estimated:  ', \
                intWithCommas(self.size), \
                'Total Models:           ', \
                intWithCommas(self.maxModels))),
                fontsize=10)
                
    
    def _plot_hist(self, key, xlabel, weight, col=None):
        '''plots histogram of parameter across models
        
        Parameters
        ----------
        key : str
            result key
        '''
        
        params = self._handle_result(key) \
                if col is None else self._handle_result(key, column=col)
        if weight is not None:
            weights = self._handle_result(weight)
        else:
            weights = None    
        
        assert(params.ndim == 1)
        assert len(params) == self.size
        
        plt.hist(params, 30, \
            weights = weights, color='0.75')
    
        plt.xlabel(xlabel, fontsize=15)
        
        plt.ylabel('Frequency', fontsize=15)
        
        xlim = plt.xlim()    
        xpad = 0.1*(xlim[1] - xlim[0])
        plt.xlim(xlim[0]-xpad, xlim[1]+xpad)
        
        plt.tick_params(axis='both', labelsize=15, \
                        top='off', right='off')
        
        plt.text(x=plt.xlim()[1]*1./2, y=plt.ylim()[1]*10.5/10, \
                s=('{0}{1}\n{2}{3}'.format(\
                'Models Estimated:  ', \
                intWithCommas(self.size), \
                'Total Models:           ', \
                intWithCommas(self.maxModels))),
                fontsize=10)        
        
        
    def format_results(self, best=False, weight_key='posterior'):
        '''summarizes regression coefficients and model fit
        
        Parameters
        ----------
        best : bool
            if True, formats results for best model
            if False, formats results for average over models
        
        Returns
        -------
        string : str
            formatted string representation of results
        '''
        model = self.best_model(weight_key)
        weight = '{0:.3f}'.format(self._handle_result(weight_key, index=model))
        weight = 'Weight ({}): '.format(weight_key) + weight if best else ''
        
        if best:
            coeff = self._handle_result('params', index=model)
            r2 = self._handle_result('par_rsquared', index=model)
        else:
            coeff = self.average('params',weight=weight_key)
            r2 = self.average('par_rsquared',weight=weight_key)
        
        # Header
        string = 'Best Model: {}'.format(model) if best else 'Bayesian Model Average'
        string += '\n{}'.format(weight)
        string += '' if best else '{} of {} Models Considered'.format(\
                        self.size, self.maxModels)
        string += '\n\nVariable Coeff Partial-R2\n' + '-'*25 + '\n'
        
        # Table
        for i in xrange(len(coeff)):
            string += 'B{0:<8}{1:>6}{2:>9}\n'.format(\
            i,'{0:.3f}'.format(coeff[i]),'{0:.3f}'.format(r2[i]))
        
        # Footer
        string += '-'*25 + '\nTotal R-Squared: {0:0.3f}'.format(r2.sum())
        
        return string
        
    def summary(self,weight='posterior'):
        '''prints summary of results
        '''
        
        print("\n\n")
        print(self.format_results(best=True,weight_key=weight))
        print("\n\n")
        print(self.format_results(weight_key=weight))
        
def intWithCommas(x):
    """return `x` formatted as long int with commas
    
    Parameters
    ----------
    x : int
        integer to be converted
    
    Returns
    -------
    strX : str
        string rep of x
    
    Example
    -------
    x = 1003949 would return '1,003,949'
	
    """

    if type(x) not in [type(0), type(0L)]:
	
        raise TypeError("Parameter must be an integer.")
		
    if x < 0:
	
        return '-' + intWithCommas(-x)
		
    result = ''
	
    while x >= 1000:
	
        x, r = divmod(x, 1000)
		
        result = ",%03d%s" % (r, result)
		
    return "%d%s" % (x, result)
        
# Test Code

if __name__ == '__main__':
    
    # Local Modules
    from Models import *
    from ModelSpace import *

    np.random.seed(1234)
    random.seed(1234)
    import time
    
    np.set_printoptions(precision=3, suppress=True)

    X = 10*np.random.randn(100, 5)

    X = np.hstack((X, X + 3*np.random.randn(100,5),X + 3*np.random.randn(100,5)))
    
    X= np.hstack((X, np.random.randn(100, 5)))

    e = 30*np.random.randn(100,1)

    B = np.array([1.5, 1, 2, 1, 2, 1]).reshape((6,1))

    Y = np.dot(sm.add_constant(X[:,:5], prepend=True), B) + e

    with ModelSpace(Y,X) as model:
        model.set_regressors(3,7)
        model.set_prior_type('collinear adjusted dilution')
        
        start = time.time()
        mcmc = pMCMC(visits=10000, burn=1000, thin=2, seed=1234, threads=4)
        print("Chain took {} seconds to run".format(time.time() - start))
        trace = Trace(mcmc)
        trace.summary()
        plt.figure(1)
        trace.plot('posterior')
        plt.figure(2)
        trace.plot('rsquared_adj', weight='posterior')
        plt.figure(3)
        trace.plot('params', weight='posterior', col=1)
        plt.figure(4)
        trace.plot('par_rsquared', weight='posterior', col=1)
        plt.show()        
        
#        model.set_prior_typ(prior_type='uniform')
#        mcmc = pMCMC(visits=10000, burn=1000, thin=2, seed=1234, threads=4)
#        trace = Trace(mcmc)
#        trace.summary()
#
#        model.set_attributes({'posterior':0, 'prior':0, 'par_rsquared':1, 'bse':1, 'bic':0, 
#                       'aic':0, 'rsquared':0, 'rsquared_adj':0, 'params':1})
#                       
#        sample = pRandomSample(draws=1000, seed=1234, threads=4)
#        trace = Trace(sample)
#        trace.summary()
        