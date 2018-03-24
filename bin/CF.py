import numpy as np
from scipy.spatial.distance import pdist, squareform


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


#--nan imputation by column mean
#Return imputed matrix, column mean
def imputation(matrix):
    
    #Compute column mean
    nMean = np.nanmean(matrix, axis=0)

    #Find nan iloc
    naniloc = np.where(np.isnan(matrix))

    #Insert appropriate value into the matrix where is nan
    #np.take is faster than fancy indexing i.e. nMean[[1, 3, 5]]
    matrix[naniloc] = np.take(nMean, naniloc[1])

    return matrix, nMean


#--SVD
#Return user pair-wise distance matrix
def SVD(matrix, nf=10):

    #Complete SVD
    #Result shape: (nM, nM) (nN, ) (nN, nN)
    u, s, vh = np.linalg.svd(matrix)

    #Truncate SVD (use t largest singular values)
    u_t = u[:, 0:nf]

    #Distance matrix
    u_dist = squareform(pdist(u_t, 'cosine'))
    
    return u_dist


#--Find the best matched raters and make reference
#Return reference rating vec and corresponding distance vec
def reference(dist_target, pref_nan, nRef):

    #Sort the rater by distance and remove self
    reference_rater = np.delete(np.argsort(dist_target), 0)

    #Make reference rating and distance
    reference_rating = []
    reference_dist = []
    for rater in reference_rater:

        #Skip nan
        if np.isnan(pref_nan[rater, n]): continue
        
        #Acquire only nRef references
        if len(reference_rating) == nRef: break

        reference_rating.append(pref_nan[rater, n])
        reference_dist.append(dist_target[rater])

    return reference_rating, reference_dist


#--CF prediction of the left out
#m, n specify the left out rating
#mode changes the way prediction is computed
#Return predicted score
def CF(pref_nan, m, n, nRef=10, mode=0):

    #Mask the pref_nan to acquire the training data
    pref_train = pref_nan.copy()
    pref_train[m, n] = np.nan

    #Impute nan with column mean 
    pref_train, nMean = imputation(pref_train)

    #If mode 1, substract col mean from pref
    if mode == 1: pref_train -= nMean

    #Perform SVD for user distance matrix
    u_dist = SVD(pref_train, nf=20)

    #Sort, remove self, and find the best matched raters and their ratings
    reference_rating, reference_dist = reference(u_dist[m, :], pref_nan, nRef)

    #Prediction
    #Subtract column mean to see the prediction of personal preference
    #Dist as weight -> transform back to -1 to 1
    computation = {
        0: np.mean(reference_rating) - nMean[n],
        1: np.mean(reference_rating) - nMean[n],
        2: np.dot(np.array(reference_rating) - nMean[n], -(np.array(reference_dist) - 1))
    }
    prediction = computation[mode]

    return prediction


#--Item similarity prediction of the left out (with an extra similarity matrix)
def itemSimF(matrix, pref_nan, m, n, nRef=10):

    #Get user distance matrix
    u_dist = squareform(pdist(matrix, 'cosine'))




'''
------------------------------------------------------------
Data
------------------------------------------------------------
'''
#--Preprocessing
pref_nan, nM, nN, isnan_inv, gameRatedByRater = preprocessing()

#How many raters rate each game
print('Number of raters per game:\n', np.sum(isnan_inv, axis=0))

#How many games rated each rater
print('Number of games rated per rater:\n', np.sum(isnan_inv, axis=1))

#Total num of rating (!= nM * nN)
nMN = len(np.where(isnan_inv)[0])
print('Total number of ratings:\n', nMN)




'''
------------------------------------------------------------
Model
------------------------------------------------------------
'''
#--Leave-one-out prediction
#CF mode
mode = 2
nMean = np.broadcast_to(np.nanmean(pref_nan, axis=0), (nM, nN))

#Operation
predictions_nan = np.full(shape=pref_nan.shape, fill_value=np.nan)
for m in np.arange(nM):
    for n in gameRatedByRater[m]:
        predictions_nan[m, n] = CF(pref_nan, m, n, nRef=10, mode=mode)

#Take non-nan entries and makes into long-form by [isnan_inv] slicing
#Also subtract column mean for pref matrix
predictions = predictions_nan[isnan_inv]
pref = pref_nan[isnan_inv] - nMean[isnan_inv]


#--Test MSE
mse = np.nansum(np.square(predictions - pref) / nMN)
cor = np.corrcoef(predictions, pref)
print('Test MSE: ', mse)
print('Test correlation: ', cor[0, 1])


#--Benchmark
#Column mean prediction MSE
mse_nMean = np.nansum(np.square(0 - pref) / nMN)
cor_nMean = np.corrcoef(np.broadcast_to(0, nMN), pref)
print('Column mean MSE: ', mse_nMean)
