import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


#--Matrix from survey ()
emb_triplet = np.genfromtxt(r'..\data\tste_embedding_25.csv', delimiter=',', skip_header=0)
print(sim_triplet.shape)


#--Matrix from text (300 features)
emb_review_raw = pd.read_csv(r'..\data\core_vec.csv')
emb_review = emb_review_raw.drop('id', axis=1).groupby('coreId').mean().as_matrix()
print(emb_review.shape)


#--Compute pairwise distance
dist_triplet = pdist(emb_triplet, 'cosine')
print(dist_triplet.shape)

dist_review = pdist(emb_review, 'cosine')
print(dist_review.shape)


#--Correlation
np.corrcoef(dist_triplet, dist_review)
