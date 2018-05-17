import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from util import preprocessing, scatter, recLoo, deMean, multiImplement


'''
------------------------------------------------------------
Preference data
------------------------------------------------------------
'''
#--Data description
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
Item similarity
------------------------------------------------------------
'''
#--Matrix from survey (25 features)
emb_triplet = np.genfromtxt(r'..\data\tste_embedding_25.csv', delimiter=',', skip_header=0)
print('sim_triplet shape: ', emb_triplet.shape)


#--Matrix from text (300 features)
emb_review_raw = pd.read_csv(r'..\data\core_vec.csv')
emb_review = emb_review_raw.drop('id', axis=1).groupby('coreId').mean().as_matrix()
print('emb_review shape: ', emb_review.shape)


#--Compute pairwise distance
dist_triplet = pdist(emb_triplet, 'cosine')
dist_review = pdist(emb_review, 'cosine')


#--Triplet and review correlation
np.corrcoef(dist_triplet, dist_review)

#Graphing
scatter([dist_triplet, dist_review], ['dist_triplet', 'dist_review'])




'''
------------------------------------------------------------
Component functions
------------------------------------------------------------
'''
#--Find the best matched items and make reference
#Return reference rating vec and corresponding distance vec
def reference_byItem(dist_target, pref_nan, pref_train, m, nRef):

    #Sort the item by distance and remove self
    reference_item = np.delete(np.argsort(dist_target), 0)

    #Make reference rating and distance
    reference_rating = []
    reference_dist = []
    for item in reference_item:

        #Skip nan
        if np.isnan(pref_nan[m, item]): continue
        
        #Acquire only nRef references
        if len(reference_rating) == nRef: break

        reference_rating.append(pref_train[m, item])
        reference_dist.append(dist_target[item])

    return reference_rating, reference_dist


#--Score prediction of the left out
#m, n specify the left out rating
#Return predicted score
def cRec(pref_nan, v_dist, m, n, nRef, mode):

    #Mask the pref_nan to acquire the training data
    pref_train = pref_nan.copy()
    pref_train[m, n] = np.nan

    #Substract row and column effects from pref
    pref_train = deMean(pref_train)[0]

    #Sort, remove self, and find the best matched raters and their ratings
    reference_rating, reference_dist = reference_byItem(v_dist[n, :], pref_nan, pref_train, m, nRef)

    #Prediction
    #Dist as weight -> transform back to -1 to 1
    computation = {
        '0': np.mean(reference_rating),
        '1': np.dot(np.array(reference_rating), -(np.array(reference_dist) - 1))
    }
    prediction = computation[mode]

    return prediction




'''
------------------------------------------------------------
Models
------------------------------------------------------------
'''
#--Subtract column and row effects for pref matrix and makes it long-form
prefs = deMean(pref_nan)[0][isnan_inv]


#--Leave-one-out cRec implementation
def implementation_c(nRef, graph=False):
    
    #Parameters
    nRef, mode = (nRef, '0')

    #Prediction
    predictions_c = recLoo(recFunc=cRec, dist=squareform(dist_triplet), nRef=nRef, mode=mode)

    #Evaluation
    mse_c = np.sum(np.square(predictions_c - prefs) / nMN)
    cor_c = np.corrcoef(predictions_c, prefs)
    print('-' * 60)
    print('CRec mode {} (reference = {})'.format(mode, nRef))
    print('MSE =', mse_c)
    print('Correlation =', cor_c[0, 1])

    #Graphing
    if graph: scatter([prefs, predictions_c], ['prefs', 'predictions_c'])

    #Return the predicted value
    return predictions_c, cor_c[0, 1]

#Implement
predictions_c, _ = implementation_c(5, graph=True)

#Implement with different numbers of reference
multiImplement(np.arange(1, 11), implementation_c, 'Content-based')