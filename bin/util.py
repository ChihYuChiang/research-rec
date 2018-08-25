import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import tensorflow as tf
import logging

'''
------------------------------------------------------------
Generic functions
------------------------------------------------------------
'''
#--Universal container
class UniversalContainer():

    def listData(self):
        data = [(item, getattr(self, item)) for item in dir(self) if not callable(getattr(self, item)) and not item.startswith("__")]
        for item in data: print(item)

    def listMethod(self):
        print([item for item in dir(self) if callable(getattr(self, item)) and not item.startswith("__")])


#--Setting container
class SettingContainer(UniversalContainer):

    def __init__(self, **kwarg):
        for key, value in kwarg.items():
            setattr(self, key, value)


#--Flatten list (recursive)
#Parameter: l, a list
#Return: a flattened list as a generator
def flattenList(l):
    import collections

    for el in l:
        if isinstance(el, collections.Sequence) and not isinstance(el, (str, bytes)):
            yield from flattenList(el)
        else:
            yield el


#--Element-wise list operation
#Return: operated list
def listEWiseOp(op, l):
    return list(map(op, l))


#--Graphing 2-D scatter plot
#With distribution and linear fitting line
def scatter(vectors, names):

    sns.set(color_codes=True)

    df = pd.DataFrame({
        names[0]: vectors[0],
        names[1]: vectors[1]
    })

    g = sns.jointplot(x=names[0], y=names[1], data=df, color="m", kind="reg", scatter_kws={"s": 10})
    plt.show()


#--Remove row and column effect
#Acquire row and column effect
def getMean(matrix):

    #Compute row and column effects
    mMean = np.nanmean(matrix, axis=1) - np.mean(np.nanmean(matrix, axis=1))
    nMean = np.nanmean(matrix, axis=0) - np.mean(np.nanmean(matrix, axis=0))

    #Deal with empty row/column (all nan)
    mMean[np.where(np.isnan(mMean))] = np.nanmean(matrix)
    nMean[np.where(np.isnan(nMean))] = np.nanmean(matrix)

    return mMean, nMean

#Return matrix with row and column effects removed (hopefully)
def deMean(matrix_in, mMean=[], nMean=[]):

    #If not provided, use the input matrix to calculate the means
    if len(mMean) == 0 or len(nMean) == 0: mMean, nMean = getMean(matrix_in)

    #Make a hard copy to avoid changing the original matrix in the function
    matrix_out = np.copy(matrix_in)

    #Compute new matrix removed the effects
    matrix_out -= (np.reshape(nMean, (1, len(nMean))) + np.reshape(mMean, (len(mMean), 1)))

    return matrix_out


#--Evaluate model with mse, cor, and graphing
def evalModel(predictions, truth, nMN, title, graph, logger=None):

    #Description
    mse = np.sum(np.square(predictions - truth)) / nMN
    cor = np.corrcoef(predictions, truth)[0, 1]
    rho, _ = sp.stats.spearmanr(predictions, truth)

    output = logger if logger else print
    output('-' * 60)
    output(title)
    output('MSE = {}'.format(mse))
    output('Correlation = {}'.format(cor))
    output('RankCorrelation = {}'.format(rho))

    #Graphing
    if graph: scatter([truth, predictions], ['truth', 'predictions'])

    return mse, cor, rho


#--Acquire ids of a k-fold training testing set
def kFold(k, nMN, seed=1):

    #Reset the seed
    np.random.seed(seed=seed)

    #The indice to be selected
    rMN = np.arange(nMN)
    np.random.shuffle(rMN)

    #Indicator
    #To make sure the distribution is as evenly as possible
    ind = abs(nMN - (nMN // k + 1) * (k - 1) - (nMN // k + 1)) < abs(nMN - (nMN // k) * (k - 1) - (nMN // k))

    #Series id based on k
    anchor = np.arange(k) * (nMN // k + ind)
    
    #Acquire the training and testing set ids
    id_test = [rMN[anchor[i]:(anchor[i + 1] if i + 1 != len(anchor) else None)] for i in range(len(anchor))]
    id_train = [np.setdiff1d(rMN, id_test[i]) for i in range(len(id_test))]

    #Deal with 1 fold (using entire dataset to train and test)
    if k == 1: id_train = id_test

    return id_train, id_test
    

#--Logger
def iniLogger(loggerName, fileName, _console):

    #Use the default logger
    logger = logging.getLogger(loggerName)
    logger.setLevel(logging.DEBUG)

    #Create formatter
    formatter = logging.Formatter('%(message)s')

    #Create file handler and add to logger
    fh = logging.FileHandler('../log/{}'.format(fileName), mode='w+')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    #Create console handler and add to logger
    if _console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


#--nan imputation by total mean and adjust by column and row effects
#Return imputed matrix
def imputation(matrix, imValue=None):
    
    #Find nan iloc
    naniloc = np.where(np.isnan(matrix))

    #Insert appropriate value into the matrix where is nan
    #Impute a predefined value
    if imValue: matrix[naniloc] = imValue
        
    #Impute overall mean
    else:
        #Compute column and row effect
        mMean, nMean = getMean(matrix)

        #np.take is faster than fancy indexing i.e. nMean[[1, 3, 5]]
        matrix[naniloc] = np.nanmean(matrix) + np.take(nMean, naniloc[1]) + np.take(mMean, naniloc[0])

        #Substract mean, col and row effects from pref
        matrix = deMean(matrix, mMean, nMean)

    return matrix




'''
------------------------------------------------------------
Common functions
------------------------------------------------------------
'''
#--Data container
class DataContainer(UniversalContainer):

    def __init__(self, _preDe=False):
        self.pref_nan, self.prefs, self.nM, self.nN, self.nMN, self.isnan_inv, self.gameRatedByRater = preprocessing(description=False, _preDe=_preDe)


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
