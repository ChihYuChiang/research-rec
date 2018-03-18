import numpy as np

#--Data and shape
pref_raw = np.genfromtxt(r'../data/raw_preference.csv', delimiter=',', skip_header=1)
nN_raw = pref_raw.shape[1]


#--Column mean imputation
pref = (pref_raw[:, np.arange(0, nN_raw, 3)] + pref_raw[:, np.arange(1, nN_raw, 3)] + pref_raw[:, np.arange(2, nN_raw, 3)]) / 3

#Compute column mean
nMean = np.nanmean(pref, axis=0)

#Find indices to replace
naniloc = np.where(np.isnan(pref))

#Insert appropriate value into the matrix
#np.take is faster than fancy indexing. e.g. nMean[[1, 3, 5]]
pref[naniloc] = np.take(nMean, naniloc[1])


#--SVD
#Complete SVD
u, s, vh = np.linalg.svd(pref)
print(u.shape, s.shape, vh.shape)

#Truncated SVD (use t largest singular values)