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
import itertools
import tables

# Third Party
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pylab as plt

from ModelSpace import ModelSpace


def _get_result(cols):
    '''obtain model results

    Parameters
    ----------
    cols : list
        columns in data

    Returns
    -------
    rslt : tuple
        column-result pair
    '''

    model_space = modelcontext()
    keep = model_space.keep

    return (cols, model_space.fit(sorted(keep + cols)))

def SampleAll():
    '''samples from all of the models'''

    model_space = modelcontext()
    allowed = model_space.k
    choices = model_space.choices
    keep = model_space.keep
    K = model_space.K
    maxm = model_space.maxm
    fname = model_space.file

    cols = itertools.chain.from_iterable(iter(itertools.combinations(choices, k-len(keep)+1) \
                                          for k in allowed))

    num = len(keep) - 1 + len(choices)
    hfile = tables.open_file(fname, 'w')
    cell = tables.Atom.from_kind('float', 8)
    results = hfile.createArray(hfile.root, 'ResultArray', 
        shape=(maxm, 2 + 3*num), atom=cell)

    for i,c in enumerate(cols):
        fit = model_space.fit(sorted(keep + c))
        results[i, [0, 1]] = fit[[0, 1]]
        rlen = len(c) + len(keep) - 1
        for j in [0, 1, 2]:
            results[i, 1 + j*num + np.array(keep[1:] + c)] = fit[2 + rlen*j:2 + rlen*(j+1)]

    results[:, 0] = results[:, 0]/results[:, 0].sum()
    results[:] = results[(-results[:,0]).argsort(), :]

    results.flush()

    return hfile

def pSampleAll(threads=2):
    '''samples from all models in parallel

    Parameters
    ----------
    threads : int
        number of processes to spawn

    Returns
    -------
    sample : dict
        column-rslt object pairs
    '''

    model_space = modelcontext()
    allowed = model_space.k
    choices = model_space.choices
    keep = model_space.keep
    K = model_space.K
    maxm = model_space.maxm
    fname = model_space.file

    p = mp.Pool(threads)

    cols = itertools.chain.from_iterable(iter(itertools.combinations(choices, k-len(keep)+1) \
                                          for k in allowed))
    
    # Pooling Results
    mapped = iter(p.map(_get_result, cols))
    p.close()
    p.join()

    # Saving Results
    num = len(keep) - 1 + len(choices)
    hfile = tables.open_file(fname, 'w')
    cell = tables.Atom.from_kind('float', 8)
    results = hfile.createArray(hfile.root, 'ResultArray', 
        shape=(maxm, 2 + 3*num), atom=cell)

    for i, (c, fit) in enumerate(mapped):
        results[i, [0, 1]] = fit[[0, 1]]
        rlen = len(c) + len(keep) - 1
        for j in [0, 1, 2]:
            results[i, 1 + j*num + np.array(keep[1:] + c)] = fit[2 + rlen*j:2 + rlen*(j+1)]

    results[:, 0] = results[:, 0]/results[:, 0].sum()
    results[:] = results[(-results[:,0]).argsort(), :]

    results.flush()

    return hfile

def random_draw():
    '''draws a set of regressors at random
    
    Parameters
    ----------
    choices : array-like
        choices for the number of regressors to go
        in model
    K : int
        total number of regressors to select from
    
    Returns
    -------
    draw : list
        set of regressors
    '''

    model_space = modelcontext()
    allowed = model_space.k
    choices = model_space.choices
    keep = model_space.keep

    k = np.random.choice(allowed)
    
    cols = tuple(np.random.choice(choices, size=k-len(keep)+1, replace=False))
    
    return sorted(keep + tuple(cols))


def mcmc_draw(last_draw):
    '''moves to next model in markov chain sampler for model space
    
    Parameters
    ----------
    last_draw : list
        set of regressors from previous draw
    cache : dict
        dictionary to store regression results
    
    Returns
    -------
    draw : list
        set of regressors
    '''

    model_space = modelcontext()
    allowed = model_space.k
    choices = model_space.choices
    keep = model_space.keep
    K = model_space.K
    maxm = model_space.maxm
    fname = model_space.file
    
    width = len(keep) + len(choices)
    prev = np.zeros(width)
    prev[last_draw] = 1
    prev = prev.reshape((-1, 1))
    
    neighbors = abs(np.diag(np.ones(width)) - prev)[:, choices]
    neighbors = neighbors[:, np.any([neighbors.sum(axis=0) == i+1
                    for i in allowed], axis=0)]
    
    draw = random.choice(xrange(neighbors.shape[1]))
    
    proposal = sorted(np.arange(neighbors.shape[0])[neighbors[:, draw] == 1])
    
    return proposal    


def MCMC(visits, burn=0, thin=1, kick=0., seed=1234):
    '''markov chain monte carlo sampler for model space
    
    Parameters
    ----------
    visits : int
        number of visits in chain
    burn : int
        number of visits to burn from beginning of chain
    thin : int
        related to fraction of visits kept in chain
    kick : float
        minimum value for transition probability
    seed : int
        seed for random number
    '''

    assert (kick <= 1) & (kick >= 0)

    model_space = modelcontext()
    allowed = model_space.k
    choices = model_space.choices
    keep = model_space.keep
    K = model_space.K
    maxm = model_space.maxm
    fname = model_space.file

    if visits >= maxm:
        return SampleAll()

    np.random.seed(seed)        

    if burn >= visits:
        raise ValueError('burn must be fewer than total visits')
    if thin < 1:
        raise ValueError('thin must be an integer 1 or greater')

    # Saving Results
    num = len(keep) - 1 + len(choices)
    hfile = tables.open_file(fname, 'w')
    cell = tables.Atom.from_kind('float', 8)
    results = hfile.createArray(hfile.root, 'ResultArray', 
        shape=(visits, 2 + 3*num), atom=cell)

    # Obtaining first draw at random
    last_draw = random_draw()
    fit = model_space.fit(last_draw)
    results[0, [0, 1]] = fit[[0, 1]]
    rlen = len(last_draw) - 1
    for j in [0, 1, 2]:
        results[0, 1 + j*num + np.array(last_draw[1:])] = fit[2 + rlen*j:2 + rlen*(j+1)]

    for i in xrange(1, visits):

        print('visit {}'.format(i))

        accepted = False

        while not accepted:

            proposal = mcmc_draw(last_draw)

            fit = model_space.fit(proposal)
            
            if results[i - 1, 0] == 0:
                prob = 1
            else:
                prob = min(1, max(kick, fit[0]/results[i - 1, 0]))

            if np.random.choice([True, False], p=[prob, 1 - prob]):

                results[i, [0, 1]] = fit[[0, 1]]
                rlen = len(proposal) - 1
                for j in [0, 1, 2]:
                    results[i, 1 + j*num + np.array(proposal[1:])] = fit[2 + rlen*j:2 + rlen*(j+1)]

                last_draw = proposal

                accepted = True


    # Burning and thinning out visits in the chain
    if (burn > 0) or (thin > 1):
        results.rename('ResultArrayFull')
        
        truncated = results.copy(newname='ResultArray', start=burn, 
            stop=results.shape[0], step=thin)
        
        results[:, 0] = results[:, 0]/results[:, 0].sum()
        results[:] = results[(-results[:,0]).argsort(), :]

        results = truncated     

    # Normalizing the posterior
    results[:, 0] = results[:, 0]/results[:, 0].sum()
    results[:] = results[(-results[:,0]).argsort(), :]

    hfile.flush()

    return hfile
    
def pMCMC(visits, burn=0, thin=1, seed=1234, threads=1):
    '''parallel markov chain monte carlo sampler for model space
    
    Parameters
    ----------
    visits : int
        number of visits per thread in chain
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

    model_space = modelcontext()
    if visits >= maxm:
        return pSampleAll(threads)

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
    '''perform basic analysis of the estimated models
    
    Parameters
    ----------
    array : tables.array.Array
        a tables array
    '''
    
    def __init__(self, array):
        self.array = array

    def mean(self, key, weight=True):
        '''obtain average estimates

        Parameters 
        ----------
        key : str
            'params', 'par_rsquared', 'rsquared', 'bse'
        weight : bool
            if True (default), weights by the posterior 
            model probabilities

        Return
        ------
        mean : array-like
            the mean of the values
        '''

        if key in ['params', 'par_rsquared']:

            if weight:
                return np.average(self[key], weights=self['posterior'], axis=0)
            else:
                return np.average(self[key], axis=0)

        elif key == 'bse':

            mean_params = self.mean(key='params', weight=weight)

            if weight:
                return np.sqrt(np.average(np.square(self['bse']) + np.square(self['params']), 
                    weights=self['posterior'], axis=0) - np.square(mean_params))
            else:
                return np.sqrt(np.average(np.square(self['bse']) + np.square(self['params']), 
                    axis=0) - np.square(mean_params))

        elif key == 'rsquared':

            if weight:
                return np.average(self[key], weights=self['posterior'])
            else:
                return np.average(self[key])
        else:
            raise KeyError('key {} not found'.format(key))

    def __getitem__(self, key):

        # Number of regressors to choose from
        plen = (self.array.shape[1] - 2)/3

        if key == 'params':
            return self.array[:, 2:plen + 2]
        elif key == 'bse':
            return self.array[:, 2 + plen:2*plen + 2]
        elif key == 'par_rsquared':
            return self.array[:, 2 + 2*plen:3*plen + 2]
        elif key == 'rsquared':
            return self.array[:, 1]
        elif key == 'posterior':
            return self.array[:, 0]
        else:
            raise KeyError('key {} not found'.format(key))

    def plot_posterior(self, limit=30, fname=None, fmat='eps'):
        '''plots posterior across models
        
        Parameters
        ----------
        limit : int
            the top most models to plot
        fname : str
            filename to save the figure, default is None and
            shows the figure
        '''
        
        posterior = self['posterior'][:limit]

        plt.bar(left=range(1, len(posterior) + 1), 
                height=posterior, color='k')
        
        plt.xlabel('Model by Rank Order', fontsize=15)
        plt.ylabel('Posterior Probability', fontsize=15)
        
        plt.tick_params(axis='both', labelsize=15, \
                    top='off', right='off')
                    
        plt.text(x=plt.xlim()[1]*1./2, y=plt.ylim()[1]*8.5/10, \
                s=('{}{}\n{}{:,.0f}'.format(\
                'Models Shown:       ', min(self.array.shape[0], limit), \
                'Models Estimated:  ', \
                self.array.shape[0])),
                fontsize=10)

        if fname is None:
            plt.show()
        else:
            plt.savefig('{}.{}'.format(fname, fmat), bbox_inches='tight')
            plt.clf()
            plt.close()
                
            
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

    data = np.hstack((Y,sm.add_constant(X)))

    kwargs = {'prior_type':'collinear adjusted dilution'}
    with ModelSpace(data[:,:10], k=[3,4,5], keep=[0, 1], kwargs=kwargs) as model:
        
        ResultTable = MCMC(40, burn=0, thin=1, kick=0.01, seed=1234)
        trace = Trace(ResultTable.get_node('/ResultArray'))

        print(trace.mean(key='params'))
        print(trace.mean(key='rsquared'))
        print(trace.mean(key='bse'))
        print(trace.mean(key='par_rsquared'))

        print(trace['params'][:10])

        trace.plot_posterior()

        ResultTable.close()
        
    #     print(model.regressors)
    #     print(model.max_models)
    #     start = time.time()
    #     sample = pSampleAll(4)
    #     trace = Trace(sample)
    #     trace.summary()
    #     mcmc = pMCMC(visits=10000, burn=1000, thin=2, seed=1234, threads=4)
    #     print("Chain took {} seconds to run".format(time.time() - start))
    #     trace = Trace(mcmc)
    #     trace.summary()
    #     plt.figure(1)
    #     trace.plot('posterior')
    #     plt.figure(2)
    #     trace.plot('rsquared_adj', weight='posterior')
    #     plt.figure(3)
    #     trace.plot('params', weight='posterior', col=1)
    #     plt.figure(4)
    #     trace.plot('par_rsquared', weight='posterior', col=1)
    #     plt.show()        
        
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