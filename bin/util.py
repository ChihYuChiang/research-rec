import numpy as np

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