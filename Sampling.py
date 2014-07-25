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

from ModelSpace import ModelSpace, modelcontext


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

def _create_table(filename, tablename, groupname, numcoeff):
    '''creates table for results

    Parameters
    ----------
    filename : str
        filepath to save results
    tablename : str
        name of table to save
    groupname : str
        the group node for result storage
    numcoeff : int
        number of parameters, for calculating table's columns

    Return
    ------
    hfile : tables file object
        reference to on-disk storage of results
    '''

    fields = ['param', 'pvalue', 'bse', 'prsquared']
    hfile = tables.open_file(filename, 'a')

    dt = np.dtype([('id', 'S40'), ('obs', int), ('posterior', float), ('rsquared', float), 
        ] + [('{}{}'.format(n, i), float) for n in fields for i in xrange(1, numcoeff+1)])
    results = hfile.createTable('/{}'.format(groupname), tablename, dt, 
        createparents=True)

    return hfile

def _append_result(table, numcoeff, fit, cols):
    '''append a regression result to a table

    Parameters 
    -----------
    table : pytables Table
        regression results table for all models 
    numcoeff : int
        number of parameters
    fit : array
        regression results from single model
    cols : list
        columns in data fitted in model
    '''

    rslt = np.zeros(3 + numcoeff*4) # By default, values are 0
    rslt[3 + numcoeff:3 + numcoeff*2] = 1 # By default pvalues are 1
    rslt[[0, 1, 2]] = fit[[0, 1, 2]]
    rlen = len(cols) - 1
    for j in [0, 1, 2, 3]:
        rslt[2 + j*numcoeff + np.array(cols[1:])] = fit[3 + rlen*j:2 + rlen*(j+1)]

    table.append([(str(cols)[1:-1].replace(' ', ''),) + tuple(rslt)])

def _process_results(table):
    '''processes results

    Parameters
    ----------
    table : tables.table.Table instance
        table of all results
    '''
    tablename = table.name

    # Normalizing the Posterior Model Probability
    posterior = table.cols.posterior
    expr = tables.Expr("sum(posterior)", uservars={"posterior":posterior})
    psum = expr.eval()
    expr = tables.Expr("posterior/psum", 
        uservars={"posterior":posterior, "psum":psum})
    expr.set_output(posterior)
    expr.eval()

    # Descending Sort on Posterior Model Probability
    table.cols.posterior.createIndex(kind='full')
    sortedTable = table.copy(newname="Sorted", sortby='posterior', step=-1)
    table.remove()
    sortedTable.rename(tablename)
    sortedTable.flush()


def SampleAll(filename, tablename, groupname=''):
    '''samples from all of the models

    Parameters
    ----------
    filename : str
        filepath to save results
    tablename : str
        name of table to save
    groupname : str
        the group node for result storage

    Returns
    -------
    hfile : pytables file
        reference to on-disk storage
    '''

    model_space = modelcontext()
    allowed = model_space.k
    choices = model_space.choices
    keep = model_space.keep
    maxm = model_space.maxm

    cols = itertools.chain.from_iterable(iter(itertools.combinations(choices, k-len(keep)+2) \
                                          for k in allowed))

    num = len(keep) - 1 + len(choices)
    hfile = _create_table(filename, tablename, groupname, num)
    resultTable = getattr(hfile.get_node('/{}'.format(groupname)), tablename)

    for i,c in enumerate(cols):
        fitcols = sorted(keep + c)
        fit = model_space.fit(fitcols)
        _append_result(resultTable, num, fit, fitcols)

    _process_results(resultTable)

    hfile.flush()

    return hfile

def pSampleAll(filename, tablename, groupname='', threads=2):
    '''samples from all models in parallel

    Parameters
    ----------
    filename : str
        filepath to save results
    tablename : str
        name of table to save
    groupname : str
        the group node for result storage
    threads : int
        number of processes to spawn

    Returns
    -------
    hfile : pytables file
        reference to on-disk storage
    '''

    model_space = modelcontext()
    allowed = model_space.k
    choices = model_space.choices
    keep = model_space.keep
    maxm = model_space.maxm

    p = mp.Pool(threads)

    cols = itertools.chain.from_iterable(iter(itertools.combinations(choices, k-len(keep)+2) \
                                          for k in allowed))
    
    # Pooling Results
    mapped = iter(p.map(_get_result, cols))
    p.close()
    p.join()

    # Saving Results
    num = len(keep) - 1 + len(choices)
    hfile = _create_table(filename, tablename, groupname, num)
    resultTable = getattr(hfile.get_node('/{}'.format(groupname)), tablename)

    for i, (c, fit) in enumerate(mapped):
        fitcols = sorted(keep + c)
        _append_result(resultTable, num, fit, fitcols)

    _process_results(resultTable)

    hfile.flush()

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
    
    cols = tuple(np.random.choice(choices, size=k-len(keep)+2, replace=False))
    
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
    maxm = model_space.maxm
    
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


def MCMC(visits, filename, tablename, groupname='', 
    burn=0, thin=1, kick=0., seed=1234):
    '''markov chain monte carlo sampler for model space
    
    Parameters
    ----------
    visits : int
        number of visits in chain
    filename : str
        filepath to save results
    tablename : str
        name of table to save
    groupname : str
        the group node for result storage
    burn : int
        number of visits to burn from beginning of chain
    thin : int
        related to fraction of visits kept in chain
    kick : float
        minimum value for transition probability
    seed : int
        seed for random number

    Returns
    -------
    hfile : pytables file
        reference to on-disk storage
    '''

    assert (kick <= 1) & (kick >= 0)

    model_space = modelcontext()
    allowed = model_space.k
    choices = model_space.choices
    keep = model_space.keep
    maxm = model_space.maxm

    if visits >= maxm:
        return SampleAll(filename, tablename, groupname)

    np.random.seed(seed)        

    if burn >= visits:
        raise ValueError('burn must be fewer than total visits')
    if thin < 1:
        raise ValueError('thin must be an integer 1 or greater')

    # Saving Results
    num = len(keep) - 1 + len(choices)
    hfile = _create_table(filename, tablename, groupname, num)
    resultTable = getattr(hfile.get_node('/{}'.format(groupname)), tablename)

    # Obtaining first draw at random
    last_draw = random_draw()
    fit = model_space.fit(last_draw)
    _append_result(resultTable, num, fit, last_draw)
    last_prob = resultTable.cols.posterior[-1]

    for i in xrange(1, visits):

        accepted = False

        while not accepted:

            proposal = mcmc_draw(last_draw)

            fit = model_space.fit(proposal)

            if last_prob == 0:
                prob = 1
            else:
                prob = min(1, max(kick, fit[0]/last_prob))

            if np.random.choice([True, False], p=[prob, 1 - prob]):

                last_draw = proposal

                _append_result(resultTable, num, fit, last_draw)

                last_prob = resultTable.cols.posterior[-1]

                accepted = True


    # Burning and thinning out visits in the chain
    if (burn > 0) or (thin > 1):
        resultTable.rename('{}Full'.format(tablename))
        
        selection = resultTable.copy(newname=tablename, start=burn, 
            stop=resultTable.shape[0], step=thin)

        resultTable.remove()
        resultTable = selection

    _process_results(resultTable)

    hfile.flush()

    return hfile
    
def pMCMC(visits, filename, tablename, groupname='', threads=2, **mcargs):
    '''parallel MCMC sampler for model space
    
    Parameters
    ----------
    visits : int
        number of total visits
    filename : str
        filepath to save results
    tablename : str
        name of table to save
    groupname : str
        the group node for result storage
    burn : int
        number of total visits to burn
    thin : int
        related to fraction of visits kept in chain
    kick : float
        minimum value for transition probability
    seed : int
        seed for random number
    threads : int
        number of threads to spawn for sampling
    
    Notes
    -----
    will run a markov chain with `visits` on every thread
    specified
    
    '''    

    burn = mcargs.get('burn', 0)
    thin = mcargs.get('thin', 1)
    kick = mcargs.get('kick', 0.)
    seed = mcargs.get('seed', 1234)

    model_space = modelcontext()
    maxm = model_space.maxm

    if visits >= maxm:
        return pSampleAll(filename, tablename, groupname, threads)

    d_visits = [int(visits/threads)]*threads
    d_visits = [v+(i<visits%threads) for i,v in enumerate(d_visits)]
    d_filename = ['t{}_{}'.format(i, filename) for i in xrange(threads)]
    d_tablename = [tablename]*threads
    d_groupname = [groupname]*threads
    d_burn = [int(burn/threads)]*threads
    d_burn = [b+(i<burn%threads) for i,b in enumerate(d_burn)]
    d_thin = [thin]*threads
    d_kick = [kick]*threads
    d_seed = [seed+i for i in xrange(threads)]

    argset = zip(d_visits, d_filename, d_tablename, d_groupname, 
        d_burn, d_thin, d_kick, d_seed)
    
    p = mp.Pool(threads)
    
    jobs = [p.apply_async(MCMC, args) for args in argset]
    for j in jobs: j.wait()
    
    p.close()
    p.join()

    # Pooling Results
    hfile = tables.open_file(d_filename[0], 'a')
    resultTable = getattr(hfile.get_node('/{}'.format(groupname)), tablename)
    for name in d_filename[1:]:
        h = tables.open_file(name, 'a')
        t = getattr(h.get_node('/{}'.format(groupname)), tablename)
        resultTable.append(t[:])
        h.close()
        os.remove(name)

    _process_results(resultTable)
    hfile.close()

    os.rename('t0_{}'.format(filename), filename)
    
    return tables.open_file(filename, 'a')
  

class Trace(object):
    '''perform basic analysis of the estimated models
    
    Parameters
    ----------
    table : tables.table.Table instance 
        regression results in on-disk container
    '''
    
    def __init__(self, table):
        self.data = self._load(table)

    @staticmethod
    def _load(table):
        '''loads the data on disk into memory'''

        data = pd.DataFrame.from_records(table[:])

        index = [('fit', d) for d in ['id', 'obs', 'posterior', 'rsquared']]
        for p in ['param', 'pvalue', 'bse', 'prsquared']:
            index.extend([(p, c) for c in data.columns if p in c])

        data.columns = pd.MultiIndex.from_tuples(index)
        
        return data

    def set_colnames(self, colnames):
        '''set column names based on the names
        of variables entered into regression'''

        columns = list(self.data.columns)
        newcols = columns[:4]
        for col in ['param', 'pvalue', 'bse', 'prsquared']:
            itercols = iter(colnames)
            for c in columns:
                c0,c1 = c
                if c0 == col:
                    c1 = itercols.next()
                    newcols.append((c0,c1))

        self.data.columns = pd.MultiIndex.from_tuples(newcols)

    def mean(self, key, weight=True):
        '''obtain average estimates

        Parameters 
        ----------
        key : str
            'param', 'prsquared', 'rsquared', 'bse'
        weight : bool
            if True (default), weights by the posterior 
            model probabilities

        Return
        ------
        mean : array-like
            the mean of the values
        '''

        if key in self.data.columns:
            data = self[key]
        elif key in ['posterior', 'rsquared', 'obs']:
            data = self['fit'][key]
        else:
            raise KeyError('{} not a valid key'.format(key))

        if weight:
            return data.mul(self['fit']['posterior'], axis='index').sum()
        else:
            return data.mean()


    def __getitem__(self, key):

        return self.data[key]

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
        
        posterior = self['fit']['posterior'].ix[:limit]

        plt.bar(left=range(1, len(posterior) + 1), 
                height=posterior, color='k')
        
        plt.xlabel('Model by Rank Order', fontsize=15)
        plt.ylabel('Posterior Probability', fontsize=15)
        
        plt.tick_params(axis='both', labelsize=15, \
                    top='off', right='off')
                    
        plt.text(x=plt.xlim()[1]*1./2, y=plt.ylim()[1]*8.5/10, \
                s=('{}{}\n{}{:,.0f}'.format(\
                'Models Shown:       ', min(self.data.shape[0], limit), \
                'Models Estimated:  ', \
                self.data.shape[0])),
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
    import os
    
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
        
        if os.path.exists('results.h5'):
            os.remove('results.h5')

        print(model.maxm)
        # ResultTable = SampleAll('results.h5', 'ResultTable')
        ResultTable = pSampleAll('results.h5', 'ResultTable', threads=4)
        # ResultTable = MCMC(4000, 'results.h5', 'ResultTable', 
        #     burn=1000, thin=1, kick=0.05, seed=1234)
        # ResultTable = pMCMC(4000, 'results.h5', 'ResultTable', threads=4, 
        #     burn=1000, thin=1, kick=0.05, seed=1234)

        trace = Trace(ResultTable.root.ResultTable)
        print(trace.mean(key='obs'))
        print(trace.mean(key='param'))
        print(trace.mean(key='rsquared'))
        print(trace.mean(key='bse'))
        print(trace.mean(key='prsquared'))
        print(trace.mean(key='param')['param3'])
        print(trace.mean(key='pvalue'))

        # trace.plot_posterior()

        ResultTable.close()
        os.remove('results.h5')