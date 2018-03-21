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
#Return imputed matrix
def imputation(matrix):
    
    #Compute column mean
    nMean = np.nanmean(matrix, axis=0)

    #Find nan iloc
    naniloc = np.where(np.isnan(matrix))

    #Insert appropriate value into the matrix where is nan
    #np.take is faster than fancy indexing i.e. nMean[[1, 3, 5]]
    matrix[naniloc] = np.take(nMean, naniloc[1])

    return matrix


#--SVD
#Return user pair-wise distance matrix
def SVD(matrix, nf=10):

    #Complete SVD
    #Result shape: (nM, nM) (nN, ) (nN, nN)
    u, s, vh = np.linalg.svd(matrix)

    #Truncate SVD (use t largest singular values)
    u_t1 = u[:, 0:nf]
    u_t2 = u[:, 1:nf + 1]

    #Distance matrix
    u_dist = squareform(pdist(u_t1, 'cosine'))
    
    return u_dist


#--CF prediction based on the left out
#m, n specify the left out rating
#Return predicted score
def CF(pref_nan, m, n):

    #Mask the pref_nan to acquire the training data
    pref_train = pref_nan.copy()
    pref_train[m, n] = np.nan

    #Impute nan with column mean 
    pref_train = imputation(pref_train)

    #Perform SVD
    u_dist = SVD(pref_train, nf=20)

    #Target user row
    targetUser = u_dist[m, :]

    #Sort, remove self, and find the best matched raters
    matched = np.delete(np.argsort(u_dist[m, :]), 0)
    reference = []
    for rater in matched:
        if np.isnan(pref_nan[rater, n]): continue
        if len(reference) == 10: break
        reference.append(pref_nan[rater, n])

    #Prediction
    prediction = np.mean(reference)

    return prediction




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
predictions = np.full(shape=pref_nan.shape, fill_value=np.nan)
for m in np.arange(nM):
    for n in gameRatedByRater[m]:
        predictions[m, n] = CF(pref_nan, m, n)


#--Test MSE
mse = np.nansum(np.square(predictions - pref_nan) / nMN)
print('Test MSE: ', mse)


#--Benchmark MSEs
#Column mean prediction MSE
nMean = np.nanmean(pref_nan, axis=0)
mse_nMean = np.nansum(np.square(nMean - pref_nan) / nMN)
print('Column mean MSE: ', mse_nMean)

#All mean prediction MSE
mnMean = np.nanmean(pref_nan)
mse_mnMean = np.nansum(np.square(mnMean - pref_nan) / nMN)
print('All mean MSE: ', mse_mnMean)