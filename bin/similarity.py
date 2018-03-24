import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

sns.set(color_codes=True)


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

dist_df = pd.DataFrame({
    'dist_triplet': dist_triplet,
    'dist_review': dist_review
})


#--Correlation
np.corrcoef(dist_triplet, dist_review)

g = sns.jointplot(x="dist_triplet", y="dist_review", data=dist_df, color="m", kind="reg", scatter_kws={"s": 10})
