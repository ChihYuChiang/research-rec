import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.spatial.distance import pdist, squareform
from generic import *

'''
------------------------------------------------------------
App options and markers
------------------------------------------------------------
'''
options = SettingContainer()
options.DEBUG = False
options.PRE_DE = True
options.K_FOLD = 1

markers = SettingContainer()
markers.CURRENT_DATA = ''




'''
------------------------------------------------------------
Model expression
------------------------------------------------------------
'''
EXP = {
    '1': {
        'var': '^(._a)|c:',
        'np': ('(m_sim ** m_a).prod(axis=0)',
            '(n_sim ** n_a).prod(axis=0)',
            'm_sim_w[m, :].reshape((data.nM, 1)) @ n_sim_w[n, :].reshape((1, data.nN))'),
        'tf': ('tf.reduce_prod(m_sim ** tf.tile(m_a, [batchSize, 1, nM, nM]), axis=1)',
            'tf.reduce_prod(n_sim ** tf.tile(n_a, [batchSize, 1, nN, nN]), axis=1)',
            'tf.reshape(tf.gather_nd(m_sim_w, simIdx_m), [batchSize, nM, 1]) @ tf.reshape(tf.gather_nd(n_sim_w, simIdx_n), [batchSize, 1, nN])')
        },
    '2': {
        'var': '^(._a)|c:',
        'np': ('(m_sim ** m_a).sum(axis=0)',
           '(n_sim ** n_a).sum(axis=0)',
           'np.broadcast_to(m_sim_w[m, :].reshape((data.nM, 1)), (data.nM, data.nN)) + np.broadcast_to(n_sim_w[n, :].reshape((1, data.nN)), (data.nM, data.nN))'),
        'tf': ('tf.reduce_sum(m_sim ** tf.tile(m_a, [batchSize, 1, nM, nM]), axis=1)',
            'tf.reduce_sum(n_sim ** tf.tile(n_a, [batchSize, 1, nN, nN]), axis=1)',
            'tf.tile(tf.reshape(tf.gather_nd(m_sim_w, simIdx_m), [batchSize, nM, 1]), [1, 1, nN]) + tf.tile(tf.reshape(tf.gather_nd(n_sim_w, simIdx_n), [batchSize, 1, nN]), [1, nM, 1])')
        },
    '2n': {
        'var': '^(._a)|c:',
        'np': ('(((m_sim >= 0) * 2 - 1) * np.absolute(m_sim) ** m_a).sum(axis=0)',
           '(((n_sim >= 0) * 2 - 1) * np.absolute(n_sim) ** n_a).sum(axis=0)',
           'np.broadcast_to(m_sim_w[m, :].reshape((data.nM, 1)), (data.nM, data.nN)) + np.broadcast_to(n_sim_w[n, :].reshape((1, data.nN)), (data.nM, data.nN))'),
        'tf': ('tf.reduce_sum((tf.cast(m_sim >= 0, tf.float32) * 2 - 1) * tf.abs(m_sim) ** tf.tile(m_a, [batchSize, 1, nM, nM]), axis=1)',
           'tf.reduce_sum((tf.cast(n_sim >= 0, tf.float32) * 2 - 1) * tf.abs(n_sim) ** tf.tile(n_a, [batchSize, 1, nN, nN]), axis=1)',
           'tf.tile(tf.reshape(tf.gather_nd(m_sim_w, simIdx_m), [batchSize, nM, 1]), [1, 1, nN]) + tf.tile(tf.reshape(tf.gather_nd(n_sim_w, simIdx_n), [batchSize, 1, nN]), [1, nM, 1])')
        },
    '3': {
        'var': '^(._a)|(._b)|c:',
        'np': ('(m_b * m_sim ** m_a).sum(axis=0)',
           '(n_b * n_sim ** n_a).sum(axis=0)',
           'np.broadcast_to(m_sim_w[m, :].reshape((data.nM, 1)), (data.nM, data.nN)) + np.broadcast_to(n_sim_w[n, :].reshape((1, data.nN)), (data.nM, data.nN))'),
        'tf': ('tf.reduce_sum(tf.tile(m_b, [batchSize, 1, nM, nM]) * m_sim ** tf.tile(m_a, [batchSize, 1, nM, nM]), axis=1)',
           'tf.reduce_sum(tf.tile(n_b, [batchSize, 1, nN, nN]) * n_sim ** tf.tile(n_a, [batchSize, 1, nN, nN]), axis=1)',
           'tf.tile(tf.reshape(tf.gather_nd(m_sim_w, simIdx_m), [batchSize, nM, 1]), [1, 1, nN]) + tf.tile(tf.reshape(tf.gather_nd(n_sim_w, simIdx_n), [batchSize, 1, nN]), [1, nM, 1])')
        },
    '3n': {
        'var': '^(._a)|(._b)|c:',
        'np': ('(((m_sim >= 0) * 2 - 1) * m_b * np.absolute(m_sim) ** m_a).sum(axis=0)',
           '(((n_sim >= 0) * 2 - 1) * n_b * np.absolute(n_sim) ** n_a).sum(axis=0)',
           'np.broadcast_to(m_sim_w[m, :].reshape((data.nM, 1)), (data.nM, data.nN)) + np.broadcast_to(n_sim_w[n, :].reshape((1, data.nN)), (data.nM, data.nN))'),
        'tf': ('tf.reduce_sum((tf.cast(m_sim >= 0, tf.float32) * 2 - 1) * tf.tile(m_b, [batchSize, 1, nM, nM]) * tf.abs(m_sim) ** tf.tile(m_a, [batchSize, 1, nM, nM]), axis=1)',
           'tf.reduce_sum((tf.cast(n_sim >= 0, tf.float32) * 2 - 1) * tf.tile(n_b, [batchSize, 1, nN, nN]) * tf.abs(n_sim) ** tf.tile(n_a, [batchSize, 1, nN, nN]), axis=1)',
           'tf.tile(tf.reshape(tf.gather_nd(m_sim_w, simIdx_m), [batchSize, nM, 1]), [1, 1, nN]) + tf.tile(tf.reshape(tf.gather_nd(n_sim_w, simIdx_n), [batchSize, 1, nN]), [1, nM, 1])')
        },
    '4': {
        'var': '^(._a)|(._b)|c:',
        'np': ('(m_b * m_sim ** m_a).sum(axis=0) + np.eye(nM)',
           '(n_b * n_sim ** n_a).sum(axis=0) + np.eye(nN)',
           'np.broadcast_to(m_sim_w[m, :].reshape((data.nM, 1)), (data.nM, data.nN)) + np.broadcast_to(n_sim_w[n, :].reshape((1, data.nN)), (data.nM, data.nN))'),
        'tf': ('tf.reduce_sum(tf.tile(m_b, [batchSize, 1, nM, nM]) * m_sim ** tf.tile(m_a, [batchSize, 1, nM, nM]), axis=1) + eyeM_batch',
           'tf.reduce_sum(tf.tile(n_b, [batchSize, 1, nN, nN]) * n_sim ** tf.tile(n_a, [batchSize, 1, nN, nN]), axis=1) + eyeN_batch',
           'tf.tile(tf.reshape(tf.gather_nd(m_sim_w, simIdx_m), [batchSize, nM, 1]), [1, 1, nN]) + tf.tile(tf.reshape(tf.gather_nd(n_sim_w, simIdx_n), [batchSize, 1, nN]), [1, nM, 1])')
        },
    '4n': {
        'var': '^(._a)|(._b)|c:',
        'np': ('(((m_sim >= 0) * 2 - 1) * m_b * np.absolute(m_sim) ** m_a).sum(axis=0) + np.eye(nM)',
           '(((n_sim >= 0) * 2 - 1) * n_b * np.absolute(n_sim) ** n_a).sum(axis=0) + np.eye(nN)',
           'np.broadcast_to(m_sim_w[m, :].reshape((data.nM, 1)), (data.nM, data.nN)) + np.broadcast_to(n_sim_w[n, :].reshape((1, data.nN)), (data.nM, data.nN))'),
        'tf': ('tf.reduce_sum((tf.cast(m_sim >= 0, tf.float32) * 2 - 1) * tf.tile(m_b, [batchSize, 1, nM, nM]) * tf.abs(m_sim) ** tf.tile(m_a, [batchSize, 1, nM, nM]), axis=1) + eyeM_batch',
           'tf.reduce_sum((tf.cast(n_sim >= 0, tf.float32) * 2 - 1) * tf.tile(n_b, [batchSize, 1, nN, nN]) * tf.abs(n_sim) ** tf.tile(n_a, [batchSize, 1, nN, nN]), axis=1) + eyeN_batch',
           'tf.tile(tf.reshape(tf.gather_nd(m_sim_w, simIdx_m), [batchSize, nM, 1]), [1, 1, nN]) + tf.tile(tf.reshape(tf.gather_nd(n_sim_w, simIdx_n), [batchSize, 1, nN]), [1, nM, 1])')
        }
}




'''
------------------------------------------------------------
Common functions
------------------------------------------------------------
'''
#--Data container
class DataContainer(UniversalContainer):

    def __init__(self, _preDe=False):
        self.pref_nan, self.prefs, self.nM, self.nN, self.nMN, self.isnan_inv, self.gameRatedByRater = preprocessing(description=False, _preDe=_preDe)
        self.naniloc_inv = np.where(self.isnan_inv)
    
    def updateByNan(self):
        self.prefs, self.nM, self.nN, self.nMN, self.isnan_inv, self.gameRatedByRater = preprocessing_core(self.pref_nan, _preDe)
        self.naniloc_inv = np.where(self.isnan_inv)


#--SVD
#Return user pair-wise distance matrix
def SVD(matrix, nf=10):

    #Complete SVD
    #Result shape: (nM, nM) (nN, ) (nN, nN)
    u, s, vh = np.linalg.svd(matrix)

    #Truncate SVD (use t largest singular values)
    u_t = u[:, 0:nf]

    #Distance matrix
    u_dist = sp.spatial.distance.squareform(sp.spatial.distance.pdist(u_t, 'cosine'))

    return u_dist


#--Remove row and column effects
#Return matrix with row and column effects removed (hopefully)
def deMean(matrix_in, mMean=[], nMean=[]):

    #If not provided, use the input matrix to calculate the means
    if len(mMean) == 0 or len(nMean) == 0: mMean, nMean = getMean(matrix_in)

    #Make a hard copy to avoid changing the original matrix in the function
    matrix_out = np.copy(matrix_in)

    #Compute new matrix removed the effects
    matrix_out -= (np.reshape(nMean, (1, len(nMean))) + np.reshape(mMean, (len(mMean), 1)))

    return matrix_out


#--Preprocess data
#Raw data + preprocessing wrapper
def preprocessing(description, _preDe=False):

    #Whether to use the preliminary regression demean
    if _preDe:
        #Load data (long form)
        rowName = pd.read_csv(r'../data/raw/raw_preference.csv', ).ResponseId.unique()
        pref_raw = pd.read_csv(r'../data/res_demean.csv')

        #Produce the item-rater matrix
        pref_nan = pd.DataFrame(index=rowName, columns=range(1, 51))
        for _, r in pref_raw.iterrows():
            pref_nan.set_value(r.respondent, r.core_id, r.res)
        pref_nan = pref_nan.values.astype(np.float32)

    else:
        #Load data
        pref_raw = np.genfromtxt(r'../data/raw_preference_combined.csv', delimiter=',', skip_header=1)
        nM_raw, nN_raw = pref_raw.shape

        #Combine sub-measurements to acquire final matrix
        #Get specific rating: pref_nan[rater, game]
        pref_nan = (pref_raw[:, np.arange(0, nN_raw, 3)] + pref_raw[:, np.arange(1, nN_raw, 3)] + pref_raw[:, np.arange(2, nN_raw, 3)]) / 3

    #Preprocessing
    prefs, nM, nN, nMN, isnan_inv, gameRatedByRater = preprocessing_core(pref_nan, _preDe)

    #Data description
    if description:
        print('Number of raters per game:\n', np.sum(isnan_inv, axis=0))
        print('Number of games rated per rater:\n', np.sum(isnan_inv, axis=1))
        print('Total number of ratings:\n', nMN)

    return pref_nan, prefs, nM, nN, nMN, isnan_inv, gameRatedByRater

#Return processed matrix, matrix shape, reversed nan index
def preprocessing_core(pref_nan, _preDe=False):

    #Get final data shape
    nM, nN = pref_nan.shape

    #Reversed nan index
    isnan_inv = np.logical_not(np.isnan(pref_nan))
    naniloc_inv = np.where(isnan_inv)

    #Find game ids of the games rated for each rater
    gameRatedByRater = [np.take(naniloc_inv[1], np.where(naniloc_inv[0] == i)).flatten() for i in np.arange(nM)]

    #Total num of rating (!= nM * nN)
    nMN = len(np.where(isnan_inv)[0])

    if _preDe:
        #It was demeaned beforehand
        prefs = pref_nan[isnan_inv]

    else:
        #Subtract column and row effects for pref matrix and makes it long-form
        prefs = deMean(pref_nan)[isnan_inv]

    return prefs, nM, nN, nMN, isnan_inv, gameRatedByRater


#--Leave-one-out implementation
#Return predicted score in long-form
def recLoo(recFunc, dist, nRef, mode, **kwargs):

    #Data
    pref_nan, prefs, nM, nN, nMN, isnan_inv, gameRatedByRater = preprocessing(description=False)

    #Operation
    predictions_nan = np.full(shape=pref_nan.shape, fill_value=np.nan)
    for m in np.arange(nM):
        for n in gameRatedByRater[m]:
            predictions_nan[m, n] = recFunc(pref_nan, dist, m, n, nRef=nRef, mode=mode, **kwargs)

    #Take non-nan entries and makes into long-form by [isnan_inv] slicing
    predictions = predictions_nan[isnan_inv]
    
    return predictions


#--Implement with different numbers of reference
def multiImplement(nRef, implementation, nRand, titleLabel):
    
    #--Perform random experiments (randomly-picked reference)
    if nRand:
        results = np.zeros((max(nRef), nRand))

        for i in nRef:
            for j in np.arange(0, nRand):
                _, cor = implementation(i, ifRand=True)
                results[i - 1, j] = cor
        
        #Draw each random result
        for j in np.arange(0, nRand): plt.plot(nRef, results[:, j], color='#f6f7eb')

        #Draw random mean
        plt.plot(nRef, np.mean(results, axis=1), color='#393e41')


    #--Real implementation
    results_true = { 'nRef': [], 'cor': [] }

    #Record result from each implementation
    for i in nRef:
        _, cor = implementation(i)
        results_true['nRef'].append(i)
        results_true['cor'].append(cor)

    #Line plot
    plt.plot(results_true['nRef'],  results_true['cor'], color='#e94f37')
    plt.title(titleLabel + ': Correlation by number of reference')
    plt.xlabel('Number of reference')
    plt.ylabel('Correlation with the real score')
    plt.show()
    plt.close()