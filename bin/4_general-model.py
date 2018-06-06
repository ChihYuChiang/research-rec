import numpy as np
from scipy.stats import t as dis_t
import matplotlib.pyplot as plt
from util import *
import warnings

DEBUG = False

#Suppress warning due to tf gather
if not DEBUG: warnings.filterwarnings("ignore")




'''
------------------------------------------------------------
Component functions
------------------------------------------------------------
'''
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




'''
------------------------------------------------------------
Ensemble model weight learning
------------------------------------------------------------
'''
#--Prepare data
#Initialize data
def gen_iniData(m_dists, n_dists, _cf):
    m_dists, n_dists = gen_ini_dist(m_dists, n_dists, _cf)
    return (m_dists, n_dists, len(m_dists) + _cf, len(n_dists))

#Compile input dataset
def gen_npDataset(m_dists, n_dists, _cf, pref_true):

    #Empty containers
    pref_trains, masks, m_sims, n_sims, truths, ms, ns = ([] for i in range(7))

    #Loop for each example
    for m in range(nM):
        for n in gameRatedByRater[m]:

            #Prepare all required inputs
            pref_train, mask = gen_pref8mask(m, n)
            m_sim, n_sim = gen_dist2sim(m_dists, n_dists, _cf, pref_train)
            truth = pref_true[m, n]

            pref_trains.append(pref_train)
            masks.append(mask)
            m_sims.append(m_sim)
            n_sims.append(n_sim)
            truths.append(truth)
            ms.append(m)
            ns.append(n)

    #Create tf dataset
    #Require np arrays; cast type explicitly
    dataset_np = {
        'pref_trains': np.stack(pref_trains).astype(np.float32),
        'masks': np.stack(masks).astype(np.bool),
        'm_sims': np.stack(m_sims).astype(np.float32),
        'n_sims': np.stack(n_sims).astype(np.float32),
        'truths': np.array(truths).astype(np.float32),
        'ms': np.array(ms).astype(np.int32),
        'ns': np.array(ns).astype(np.int32)
    }

    #Return processed dataset (saving processing, don't use generator)
    return dataset_np, len(dataset_np['ns'])


#--Learn weight (average similarity)
def gen_learnWeight(m_dists, n_dists, _cf, nRef, nEpoch, global_step, learning_rate, title, graph=False):

    #--Log
    title += '({})'.format(nRef)
    print('-' * 60)
    print(title)


    #--Initialization
    #Prepare raw data
    m_dists_processed, n_dists_processed, nMDist, nNDist = gen_iniData(m_dists, n_dists, _cf)
    dataset_np, nExample = gen_npDataset(m_dists_processed, n_dists_processed, _cf, deMean(pref_nan)[0])

    #Reset graph
    tf.reset_default_graph()


    #--Variables to be learnt
    m_w = tf.Variable(np.ones([nMDist, 1, 1]), name='m_w', dtype=tf.float32)
    n_w = tf.Variable(np.ones([nNDist, 1, 1]), name='n_w', dtype=tf.float32)
    b0 = tf.Variable(0.0, name='b0', dtype=tf.float32)
    b1 = tf.Variable(1.0, name='b1', dtype=tf.float32)


    #--Input
    #Input placeholders
    pref_train_ds = tf.placeholder(tf.float32, [nExample, nM, nN], name='pref_train')
    mask_ds = tf.placeholder(tf.bool, [nExample, nM, nN], name='mask')
    m_sim_ds = tf.placeholder(tf.float32, [nExample, nMDist, nM, nM], name='m_sim')
    n_sim_ds = tf.placeholder(tf.float32, [nExample, nNDist, nN, nN], name='n_sim')
    pref_true_ds = tf.placeholder(tf.float32, [nExample], name='pref_true')
    m_id_ds = tf.placeholder(tf.int32, [nExample], name='m_id')
    n_id_ds = tf.placeholder(tf.int32, [nExample], name='n_id')

    #Dataset and iterator
    dataset = tf.data.Dataset.from_tensor_slices((
        pref_train_ds, mask_ds,
        m_sim_ds, n_sim_ds, pref_true_ds, m_id_ds, n_id_ds
    ))
    dataset = dataset.repeat(nEpoch)
    iterator = dataset.make_initializable_iterator()
    pref_train, mask, m_sim, n_sim, pref_true, m_id, n_id = iterator.get_next()


    #--Operations
    #Intermediate
    m_sim_w = tf.reduce_prod(m_sim ** tf.tile(m_w, [1, nM, nM]), axis=0)
    n_sim_w = tf.reduce_prod(n_sim ** tf.tile(n_w, [1, nN, nN]), axis=0)
    mn_sim = tf.matmul(tf.reshape(m_sim_w[m_id, :], [nM, 1]), tf.reshape(n_sim_w[n_id, :], [1, nN]))
    mn_sim_mask = tf.reshape(tf.boolean_mask(mn_sim, mask), [-1])
    pref_train_mask = tf.reshape(tf.boolean_mask(pref_train, mask), [-1])

    if nRef != -1:
        _, refIdx = tf.nn.top_k(mn_sim_mask, k=nRef, sorted=True)
        refIdx = refIdx[:nRef]

    #Prediction
    if nRef == -1:
        pred = b0 + b1 * (tf.reduce_sum(pref_train_mask * mn_sim_mask) / tf.reduce_sum(mn_sim_mask))
    else:
        pred = b0 + b1 * (tf.reduce_sum(tf.gather(pref_train_mask * mn_sim_mask, refIdx)) / tf.reduce_sum(tf.gather(mn_sim_mask, refIdx)))
    
    #Cost (SE)
    cost = (pred - pref_true) ** 2


    #--Optimizer, initializer, saver
    opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=5)


    #--Session
    #For cost graphing
    costs = []

    with tf.Session() as sess:

        #Initialize vars/restore from checkpoint
        if global_step == 0: sess.run(init)
        else: saver.restore(sess, './../data/checkpoint/emb-update-{}'.format(global_step))

        #Initialize an iterator over the dataset
        #(has been repeated nEpoch times)
        sess.run(iterator.initializer, feed_dict={
            pref_train_ds: dataset_np['pref_trains'],
            mask_ds: dataset_np['masks'],
            m_sim_ds: dataset_np['m_sims'],
            n_sim_ds: dataset_np['n_sims'],
            pref_true_ds: dataset_np['truths'],
            m_id_ds: dataset_np['ms'],
            n_id_ds: dataset_np['ns']
        })

        #Loop over number of epochs
        for ep in range(nEpoch):

            #Get track of the cost of each epoch
            cost_epoch = 0

            #Loop over number of example
            for _ in range(nExample):

                #Run operation
                _, cost_example = sess.run([opt, cost])

                #Tally the cost
                cost_epoch += cost_example

            if ep % 10 == 0: #For text printing
                print('Cost after epoch %i: %f' % (ep, cost_epoch))
            if ep % 1 == 0: #For graphing
                costs.append(cost_epoch)
        
        #Graphing the change of the costs
        if graph:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per batch)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
            plt.close()

        #Output
        output = sess.run({'m_w': m_w, 'n_w': n_w, 'b': [b0, b1]})
        print('m weight:', list(output['m_w'].flatten()))
        print('n weight:', list(output['n_w'].flatten()))
        print('b:', list(output['b']))

        saver.save(sess, './../data/checkpoint/gen_weight_{}'.format(title), global_step=global_step + nEpoch)

        #Format for plugging into the model function
        output['m_dists'] = m_dists
        output['n_dists'] = n_dists
        output['nRef'] = nRef
        output['title'] = title

        return output


#--Train model
#u_dist_person  u_dist_sat  u_dist_demo  dist_triplet  dist_review
output_1 = gen_learnWeight(m_dists=[u_dist_sat], n_dists=[], _cf=True, nRef=-1, global_step=0, nEpoch=100, learning_rate=0.01, title='CF+satisfaction')
output_2 = gen_learnWeight(m_dists=[u_dist_sat], n_dists=[], _cf=True, nRef=20, global_step=0, nEpoch=100, learning_rate=0.01, title='CF+satisfaction')




'''
------------------------------------------------------------
Model

- Read in the data and functions in 1. and 2. by hand.
------------------------------------------------------------
'''
#--Average similarity
def gen_model(nRef, m_dists, n_dists, m_w, n_w, b, title, graph=False):
    
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
            predictions_nan[m, n] = b[0] + b[1] * (np.sum((pref_train[mask] * mn_sim[mask])[refIdx]) / np.sum(mn_sim[mask][refIdx]))

            if DEBUG: print('ref_rating', pref_train[mask][refIdx])
            if DEBUG: print('ref_sim', mn_sim[mask][refIdx])
            if DEBUG: print('prediction', predictions_nan[m, n])
            if DEBUG: print(m, n)
            if DEBUG: return ["", ""]

    #Take non-nan entries and makes into long-form by [isnan_inv] slicing
    predictions_gen = predictions_nan[isnan_inv]

    #Evaluation
    mse_gen, cor_gen = evalModel(predictions_gen, prefs, nMN, title=title, graph=graph)

    #Return the predicted value
    return predictions_gen, cor_gen

DEBUG = False
#Use nRef = -1 to employ all cells other than self
#u_dist_person  u_dist_demo  dist_triplet  dist_review
predictions_gen, cor_gen = gen_model(nRef=-1, m_dists=[], n_dists=[dist_review], m_w=[3.6908505], n_w=[14.863923], b=[-0.29466018, 1.044427], title='General model (b0+b1R)', graph=True)

#Pipeline input
predictions_gen, cor_gen = gen_model(**output_1)
predictions_gen, cor_gen = gen_model(**output_2)

