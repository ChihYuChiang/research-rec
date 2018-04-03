import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#--Preprocess data
#Return processed matrix, matrix shape, reversed nan index
def preprocessing():

    #Load data
    pref_raw = np.genfromtxt(r'../data/raw_preference2.csv', delimiter=',', skip_header=1)
    nM_raw, nN_raw = pref_raw.shape


    #Combine sub-measurements to acquire final matrix
    #Get specific rating: pref_nan[rater, game]
    pref_nan = (pref_raw[:, np.arange(0, nN_raw, 3)] + pref_raw[:, np.arange(1, nN_raw, 3)] + pref_raw[:, np.arange(2, nN_raw, 3)]) / 3

    #Get final data shape
    nM, nN = pref_nan.shape

    #Reversed nan index
    isnan_inv = np.logical_not(np.isnan(pref_nan))
    naniloc_inv = np.where(isnan_inv)

    #Find game ids of the games rated for each rater
    gameRatedByRater = [np.take(naniloc_inv[1], np.where(naniloc_inv[0] == i)).flatten() for i in np.arange(nM)]

    return pref_nan, nM, nN, isnan_inv, gameRatedByRater


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


#--Leave-one-out implementation
#Return predicted score in long-form
def recLoo(recFunc, dist, nRef, mode):

    #Data
    pref_nan, nM, nN, isnan_inv, gameRatedByRater = preprocessing()

    #Operation
    predictions_nan = np.full(shape=pref_nan.shape, fill_value=np.nan)
    for m in np.arange(nM):
        for n in gameRatedByRater[m]:
            predictions_nan[m, n] = recFunc(pref_nan, dist, m, n, nRef=nRef, mode=mode)

    #Take non-nan entries and makes into long-form by [isnan_inv] slicing
    predictions = predictions_nan[isnan_inv]
    
    return predictions


#--Remove row and column effect
#Return matrix with effects removed (hopefully)
def deMean(matrix_in):

    #Make a hard copy to avoid changing the original matrix in the function
    matrix_out = np.copy(matrix_in)

    #Compute row and column effects
    nMean = np.nanmean(matrix_out, axis=0) - np.mean(np.nanmean(matrix_out, axis=0))
    mMean = np.nanmean(matrix_out, axis=1) - np.mean(np.nanmean(matrix_out, axis=1))
    
    #Compute new matrix removed the effects
    matrix_out -= (np.reshape(nMean, (1, len(nMean))) + np.reshape(mMean, (len(mMean), 1)))

    return matrix_out, nMean, mMean