import numpy as np
from scipy.spatial.distance import pdist, squareform
from itertools import chain


#--Data and shape
pref_raw = np.genfromtxt(r'../data/raw_preference2.csv', delimiter=',', skip_header=1)
nM_raw, nN_raw = pref_raw.shape

#Combine sub-measurements
pref_nan = (pref_raw[:, np.arange(0, nN_raw, 3)] + pref_raw[:, np.arange(1, nN_raw, 3)] + pref_raw[:, np.arange(2, nN_raw, 3)]) / 3

#Get final shape
nM, nN = pref_nan.shape

#Nan index
isnan_inv = np.logical_not(np.isnan(pref_nan))

#How many raters rate each game
print('Rater per game:\n', np.sum(isnan_inv, axis=0))

#How many games rated each rater
print('Game per rater:\n', np.sum(isnan_inv, axis=1))


#--Find game ids of the games are rated for each rater
naniloc_inv = np.where(isnan_inv)

temp = [np.take(naniloc_inv[1], np.where(naniloc_inv[0] == i)).flatten() for i in np.arange(nM)]

#Get specific rating: [Rater][game]
pref_nan[0][3]


#--Leave-one-out cross validation
#Select games as the test target
[np.random.choice(temp[i]) for i in np.arange(nM)]


#--Column mean imputation
#Compute column mean
nMean = np.nanmean(pref_nan, axis=0)

#Find nan iloc
naniloc = np.where(np.isnan(pref_nan))

#Insert appropriate value into the matrix where is nan
#np.take is faster than fancy indexing. e.g. nMean[[1, 3, 5]]
pref = pref_nan.copy()
pref[naniloc] = np.take(nMean, naniloc[1])


#--SVD
#Complete SVD
u, s, vh = np.linalg.svd(pref)
print(u.shape, s.shape, vh.shape)

#Truncated SVD (use t largest singular values)
u_t1 = u[:, 0:20]
u_t2 = u[:, 1:21]

u_dist1 = squareform(pdist(u_t1, 'cosine'))
u_dist1[0, :]
