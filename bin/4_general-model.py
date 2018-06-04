import numpy as np
from scipy.stats import t as dis_t
import matplotlib.pyplot as plt
from util import *

DEBUG = False


'''
------------------------------------------------------------
Component functions
------------------------------------------------------------
'''
#--Initialize model
#Initialize distance
def gen_ini_dist(m_dists, n_dists, _cf):

    #Deal with empty
    #Create distance matrix, diagonal = 0, others = 2
    #(distance range from 0 to 2)
    if len(m_dists) == 0 and _cf == False: m_dists = [-(np.eye(nM) * 2) + 2]
    if len(n_dists) == 0: n_dists = [-(np.eye(nN) * 2) + 2]

    #Transform input into proper format
    m_dists = np.stack(m_dists) if len(m_dists) > 0 else np.empty((0, 0)) #Deal with CF only
    n_dists = np.stack(n_dists)

    return (m_dists, n_dists)

#Initialize weight
def gen_ini_w(m_w, n_w):

    #Deal with empty
    if len(m_w) == 0: m_w = [1]
    if len(n_w) == 0: n_w = [1]

    #Transform input into proper format
    m_w = np.array(m_w).reshape((len(m_w), 1, 1))
    n_w = np.array(n_w).reshape((len(n_w), 1, 1))

    return (m_w, n_w)

#Prepare pref_train for a particular target cell as rating reference
#Prepare a mask masking target self and all nan cells
def gen_pref8mask(m, n):

    #Pref
    #Mask the pref_nan to acquire training (reference) data
    pref_train = pref_nan.copy()
    pref_train[m, n] = np.nan

    #Impute nan with total mean and adjust by column and row effects
    pref_train = imputation(pref_train)

    #Mask
    #Remove self from the matrix 
    isnan_inv_mn = isnan_inv.copy()
    isnan_inv_mn[m, n] = False

    return (pref_train, isnan_inv_mn)

#Compute CF (if needed), transform distance to similarity
def gen_dist2sim(m_dists, n_dists, _cf, pref_train):

    #If we are going to include CF dist
    #Each rating uses the corresponding cf similarity
    if _cf:
        #Get CF dist and append to m_dists
        #nf = number of features used in SVD
        m_dist_cf = SVD(pref_train, nf=20)
        m_dist_cf = m_dist_cf.reshape((1, ) + m_dist_cf.shape)
        m_dists_processed = np.concatenate((m_dists, m_dist_cf), axis=0) if len(m_dists) > 0 else np.stack(m_dist_cf)
    
    else: m_dists_processed = m_dists

    #Flip distances (2 to 0) to similarities (0 to 1)
    #https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/generated/scipy.spatial.distance.cosine.html
    m_sim = 1 - m_dists_processed / 2
    n_sim = 1 - n_dists / 2

    if DEBUG: print('m_sim', m_sim)
    if DEBUG: print('n_sim', n_sim)

    return (m_sim, n_sim)


#--Ensemble model weighting
#Average similarity
def ensembleWeight_as(distStack, prefs, nRef, nEpoch=2000, graph=False):
    
    #Initialization
    #Minimize MSE
    tf.reset_default_graph()1
    learning_rate = 0.01
    distStack = tf.Variable(np.stack([tt1, tt2]), name='test', dtype=tf.float32)
    w = tf.Variable(np.ones((2, 1, 1)), name='weight', dtype=tf.float32)
    dist = tf.reduce_sum(distStack * w, axis=0)

    s, i = tf.nn.top_k(dist, k=nRef, sorted=True)
    cat_i = tf.reshape(tf.transpose(tf.range(0, 2.0) * tf.ones((2, 2))), [2, 2, 1])
    cat_i = tf.cast(cat_i, tf.int32)
    i = tf.reshape(i, [2, 2, 1])
    ee = tf.concat([i, cat_i], axis=-1)
    result = tf.gather_nd(prefs, ee)

    prediction = tf.reduce_mean(result, axis=-1)

    # cost = tf.reduce_mean(tf.square(prediction - prefs))
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    #Training
    costs = []
    with tf.Session() as sess:
        sess.run(init)
        test = sess.run(result)
        print(test)

# tt1 = np.array([[1, 2, 3], [4, 5, 6]], dtype='float32')
# tt2 = np.array([[10, 20, 30], [60, 50, 40]], dtype='float32')
# prefs = np.array([[15, 25, 35], [1.5, None, 3.5]], dtype='float32')
# np.sum(np.stack([tt1, tt2]) * np.array([[[0.1]], [[0.5]]]), axis=0)
# ensembleWeight_as([tt1, tt2], prefs, 2)




'''
------------------------------------------------------------
Ensemble model

- Read in the data and functions in 1. and 2. by hand.
------------------------------------------------------------
'''
#--Average similarity
def ensemble_as(nRef, m_dists, n_dists, m_w, n_w, title, graph=False):
    
    #Initialize
    _cf = len(m_dists) < len(m_w) #Marker, if include cf in model
    m_dists, n_dists = gen_ini_dist(m_dists, n_dists, _cf)
    m_w, n_w = gen_ini_w(m_w, n_w)

    #Prepare an empty prediction hull
    predictions_nan = np.full(shape=pref_nan.shape, fill_value=np.nan)

    #Prediction, each cell by each cell
    for m in range(nM):
        for n in gameRatedByRater[m]:

            if DEBUG:
                if m != 2 or n != 1: continue
            
            #Prepare the reference ratings
            #Prepare the mask remove self and nan from the matrix 
            pref_train, mask = gen_pref8mask(m, n)

            #Compute CF, transform distance to similarity
            m_sim, n_sim = gen_dist2sim(m_dists, n_dists, _cf, pref_train)

            #Combine weighting and similarity
            m_sim_w = (m_sim ** m_w).sum(axis=0)
            n_sim_w = (n_sim ** n_w).sum(axis=0)

            if DEBUG: print('m_sim_w', m_sim)
            if DEBUG: print('n_sim_w', n_sim)

            #Combine two types of similarities
            mn_sim = np.matmul(m_sim_w[m, :].reshape((nM, 1)), n_sim_w[n, :].reshape((1, nN)))

            if DEBUG: print('mn_sim', mn_sim)

            #Use negative sign to reverse sort
            #Acquire only nRef references
            #(index is in flatten and nan removed)
            refIdx = np.argsort(-mn_sim[mask])[:nRef]

            #Flatten, nan removed, and make prediction based on the combined similarity
            predictions_nan[m, n] = np.sum(pref_train[mask][refIdx] * mn_sim[mask][refIdx]) / np.sum(mn_sim[mask][refIdx])

            if DEBUG: print('ref_rating', pref_train[mask][refIdx])
            if DEBUG: print('ref_sim', mn_sim[mask][refIdx])
            if DEBUG: print('prediction', predictions_nan[m, n])
            if DEBUG: print(m, n)
            if DEBUG: return ["", ""]

    #Take non-nan entries and makes into long-form by [isnan_inv] slicing
    predictions_en = predictions_nan[isnan_inv]

    #Evaluation
    mse_en, cor_en = evalModel(predictions_en, prefs, nMN, title=title + ' (reference = {})'.format(nRef), graph=graph)

    #Return the predicted value
    return predictions_en, cor_en

DEBUG = False
#Use nRef = -1 to employ all cells other than self
#u_dist_person  u_dist_demo  dist_triplet  dist_review
predictions_en, cor_en = ensemble_as(nRef=5, m_dists=[], n_dists=[dist_triplet], m_w=[], n_w=[1], title='General model (test)', graph=False)
