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
#Prepare data
def gen_preData():

    return (m_dists, n_dists, len(m_dists), len(n_dists), pref_train)

#Create graph
def gen_graph(nMDist, nNDist, nRef, learning_rate):
    tf.reset_default_graph()

    #Variables to be learnt
    m_w = tf.Variable(np.zeros([nMDist, 1, 1]), name='m_w', dtype=tf.float32)
    n_w = tf.Variable(np.zeros([nNDist, 1, 1]), name='n_w', dtype=tf.float32)

    #Input
    pref_train = tf.placeholder(tf.float32, [nM, nN], name='pref_train')
    mask = tf.placeholder(tf.bool, [nM, nN], name='mask')
    m_sim = tf.placeholder(tf.float32, [nMDist, nM, nM], name='m_sim')
    n_sim = tf.placeholder(tf.float32, [nNDist, nN, nN], name='n_sim')
    pref_true = tf.placeholder(tf.float32, [1], name='pref_true')
    m_id = tf.placeholder(tf.int32, [1], name='m_id')
    n_id = tf.placeholder(tf.int32, [1], name='n_id')
    
    #Intermediate
    m_sim_w = tf.reduce_sum(m_sim ** tf.tile(m_w, [1, nM, nM]), axis=0)
    n_sim_w = tf.reduce_sum(n_sim ** tf.tile(n_w, [1, nN, nN]), axis=0)
    mn_sim = tf.matmul(tf.reshape(m_sim_w[m_id, :], [nM, 1]), tf.reshape(n_sim_w[n_id, :], [1, nN]))
    mn_sim_mask = tf.reshape(tf.boolean_mask(mn_sim, mask), [-1])
    pref_train_mask = tf.reshape(tf.boolean_mask(pref_train, mask), [-1])
    _, refIdx = tf.nn.top_k(mn_sim_mask, k=nRef, sorted=True)
    refIdx = refIdx[:nRef]

    #Cost (MSE)
    cost = tf.reduce_sum(tf.gather(pref_train_mask * mn_sim_mask, refIdx)) / tf.reduce_sum(tf.gather(mn_sim_mask, refIdx)) - pref_true

    #Optimizer, initializer
    opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    return (cost, opt, init)

gen_graph(2, 1, 10, 0.01)


#Learn weight (average similarity)
def gen_learnWeight(data, nRef, nEpoch, learning_rate, graph=False):

    #Extract data
    m_dists, n_dists, nMDist, nNDist, pref_train = data

    #Initialize graph
    cost, opt, init = gen_graph(nMDist, nNDist, nRef, learning_rate)

    #For graphing
    costs = []

    with tf.Session() as sess:
        sess.run(init)

        for ep in range(nEpoch):
            #Get track of the cost of each epoch
            cost_epoch = 0

            for m in range(nM):
                for n in gameRatedByRater[m]:
                    _, cost_example = sess.run([opt, cost],
                        feed_dict={
                            x_ij: item[1],
                            pref_true = deMean(pref_nan)[0][m, n]
                        }
                    )

                    #Tally the cost for each batch
                    cost_epoch += cost_example

            if ep % 1 == 0: #For text printing
                print('Cost after epoch %i: %f' % (ep, cost_epoch))
            if i % 1 == 0: #For graphing
                costs.append(cost_batch)




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

            #Combine weighting and similarity (sequence of product)
            m_sim_w = (m_sim ** m_w).prod(axis=0)
            n_sim_w = (n_sim ** n_w).prod(axis=0)

            if DEBUG: print('m_sim_w', m_sim)
            if DEBUG: print('n_sim_w', n_sim)

            #Combine two types of similarities (product)
            mn_sim = np.matmul(m_sim_w[m, :].reshape((nM, 1)), n_sim_w[n, :].reshape((1, nN)))

            if DEBUG: print('mn_sim', mn_sim)

            #Use negative sign to reverse sort
            #Acquire only nRef references
            #(index is in flatten and nan removed)
            refIdx = np.argsort(-mn_sim[mask])[:nRef]

            #Flatten, nan removed, and make prediction based on the combined similarity
            predictions_nan[m, n] = np.sum((pref_train[mask] * mn_sim[mask])[refIdx]) / np.sum(mn_sim[mask][refIdx])

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
predictions_en, cor_en = ensemble_as(nRef=5, m_dists=[], n_dists=[dist_triplet, dist_review], m_w=[], n_w=[1, 1], title='General model (test)', graph=False)
