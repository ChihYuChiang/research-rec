import numpy as np
import math
import re
from scipy.stats import t as dis_t
import matplotlib.pyplot as plt
import seaborn as sns
from util import *
import warnings
import logging

#--Debuggin setting
#General logger
logger = iniLogger('GM.log', extrah) if 'logger' not in globals() else logger

#CSV logger
logger_csv = logging.getLogger('csv')
formatter = logging.Formatter('%(message)s')
fh = logging.FileHandler('../log/GM.csv', mode='w+')
fh.setFormatter(formatter)
logger_csv.addHandler(fh)

logger_csv.info('\"test tile is like this\", 35, 26, 55')

#Markers
_currentData = ''
DEBUG = False

#Suppress warning due to tf gather
if not DEBUG: warnings.filterwarnings("ignore")


'''
------------------------------------------------------------
Preference data
------------------------------------------------------------
'''
#--Preprocessing pref data
#Read from file, processing without folds
pref_nan, prefs, nM, nN, nMN, isnan_inv, gameRatedByRater = preprocessing(description=False)

#Number of example of the entire data set
nMN_whole = nMN

#Log
_currentData = 'whole'
print('-' * 60)
print('Now using the entire data set.')


#--Updating data as training and test sets
def gen_preprocessing_kFold(foldId, _marker):
    assert foldId >= 0, 'Fold ID starts from 1.'
    assert _marker in ['training', 'test'], 'Wrong marker.'

    #Fold id 1 -> 0
    foldId -= 1
    
    #Manage global directly
    global pref_nan, prefs, nM, nN, nMN, isnan_inv, gameRatedByRater
    global _currentData

    #Reset data
    pref_nan, prefs, nM, nN, nMN, isnan_inv, gameRatedByRater = preprocessing(description=False)
    naniloc_inv = np.where(isnan_inv)

    #Test set blanks training set ids
    if _marker == 'test':
        nanCell = [np.take(naniloc_inv[0], id_train[foldId]), np.take(naniloc_inv[1], id_train[foldId])]
        pref_nan[nanCell] = np.nan

    #Training set blanks test set ids
    if _marker == 'training':
        nanCell = [np.take(naniloc_inv[0], id_test[foldId]), np.take(naniloc_inv[1], id_test[foldId])]
        pref_nan[nanCell] = np.nan

    #Update global vars
    prefs, nM, nN, nMN, isnan_inv, gameRatedByRater = preprocessing_core(pref_nan)

    #Log
    _currentData = 'fold #{}/{}, set {}.'.format(foldId + 1, K_FOLD, _marker.upper())
    logger.debug('-' * 60)
    logger.debug('Now using ' + _currentData)




'''
------------------------------------------------------------
Model expression
------------------------------------------------------------
'''
EXP = {
    '1': {
        'var': '^(._a)|c:',
        'np': ('(m_sim ** m_a).prod(axis=0)',
            '(n_sim ** n_a).prod(axis=0)',
            'm_sim_w[m, :].reshape((nM, 1)) @ n_sim_w[n, :].reshape((1, nN))'),
        'tf': ('tf.reduce_prod(m_sim ** tf.tile(m_a, [batchSize, 1, nM, nM]), axis=1)',
            'tf.reduce_prod(n_sim ** tf.tile(n_a, [batchSize, 1, nN, nN]), axis=1)',
            'tf.reshape(tf.gather_nd(m_sim_w, simIdx_m), [batchSize, nM, 1]) @ tf.reshape(tf.gather_nd(n_sim_w, simIdx_n), [batchSize, 1, nN])')
        },
    '2': {
        'var': '^(._a)|c:',
        'np': ('(m_sim ** m_a).sum(axis=0)',
           '(n_sim ** n_a).sum(axis=0)',
           'np.broadcast_to(m_sim_w[m, :].reshape((nM, 1)), (nM, nN)) + np.broadcast_to(n_sim_w[n, :].reshape((1, nN)), (nM, nN))'),
        'tf': ('tf.reduce_sum(m_sim ** tf.tile(m_a, [batchSize, 1, nM, nM]), axis=1)',
            'tf.reduce_sum(n_sim ** tf.tile(n_a, [batchSize, 1, nN, nN]), axis=1)',
            'tf.tile(tf.reshape(tf.gather_nd(m_sim_w, simIdx_m), [batchSize, nM, 1]), [1, 1, nN]) + tf.tile(tf.reshape(tf.gather_nd(n_sim_w, simIdx_n), [batchSize, 1, nN]), [1, nM, 1])')
        },
    '2n': {
        'var': '^(._a)|c:',
        'np': ('(((m_sim >= 0) * 2 - 1) * np.absolute(m_sim) ** m_a).sum(axis=0)',
           '(((n_sim >= 0) * 2 - 1) * np.absolute(n_sim) ** n_a).sum(axis=0)',
           'np.broadcast_to(m_sim_w[m, :].reshape((nM, 1)), (nM, nN)) + np.broadcast_to(n_sim_w[n, :].reshape((1, nN)), (nM, nN))'),
        'tf': ('tf.reduce_sum((tf.cast(m_sim >= 0, tf.float32) * 2 - 1) * tf.abs(m_sim) ** tf.tile(m_a, [batchSize, 1, nM, nM]), axis=1)',
           'tf.reduce_sum((tf.cast(n_sim >= 0, tf.float32) * 2 - 1) * tf.abs(n_sim) ** tf.tile(n_a, [batchSize, 1, nN, nN]), axis=1)',
           'tf.tile(tf.reshape(tf.gather_nd(m_sim_w, simIdx_m), [batchSize, nM, 1]), [1, 1, nN]) + tf.tile(tf.reshape(tf.gather_nd(n_sim_w, simIdx_n), [batchSize, 1, nN]), [1, nM, 1])')
        },
    '3': {
        'var': '^(._a)|(._b)|c:',
        'np': ('(m_b * m_sim ** m_a).sum(axis=0)',
           '(n_b * n_sim ** n_a).sum(axis=0)',
           'np.broadcast_to(m_sim_w[m, :].reshape((nM, 1)), (nM, nN)) + np.broadcast_to(n_sim_w[n, :].reshape((1, nN)), (nM, nN))'),
        'tf': ('tf.reduce_sum(tf.tile(m_b, [batchSize, 1, nM, nM]) * m_sim ** tf.tile(m_a, [batchSize, 1, nM, nM]), axis=1)',
           'tf.reduce_sum(tf.tile(n_b, [batchSize, 1, nN, nN]) * n_sim ** tf.tile(n_a, [batchSize, 1, nN, nN]), axis=1)',
           'tf.tile(tf.reshape(tf.gather_nd(m_sim_w, simIdx_m), [batchSize, nM, 1]), [1, 1, nN]) + tf.tile(tf.reshape(tf.gather_nd(n_sim_w, simIdx_n), [batchSize, 1, nN]), [1, nM, 1])')
        },
    '3n': {
        'var': '^(._a)|(._b)|c:',
        'np': ('(((m_sim >= 0) * 2 - 1) * m_b * np.absolute(m_sim) ** m_a).sum(axis=0)',
           '(((n_sim >= 0) * 2 - 1) * n_b * np.absolute(n_sim) ** n_a).sum(axis=0)',
           'np.broadcast_to(m_sim_w[m, :].reshape((nM, 1)), (nM, nN)) + np.broadcast_to(n_sim_w[n, :].reshape((1, nN)), (nM, nN))'),
        'tf': ('tf.reduce_sum((tf.cast(m_sim >= 0, tf.float32) * 2 - 1) * tf.tile(m_b, [batchSize, 1, nM, nM]) * tf.abs(m_sim) ** tf.tile(m_a, [batchSize, 1, nM, nM]), axis=1)',
           'tf.reduce_sum((tf.cast(n_sim >= 0, tf.float32) * 2 - 1) * tf.tile(n_b, [batchSize, 1, nN, nN]) * tf.abs(n_sim) ** tf.tile(n_a, [batchSize, 1, nN, nN]), axis=1)',
           'tf.tile(tf.reshape(tf.gather_nd(m_sim_w, simIdx_m), [batchSize, nM, 1]), [1, 1, nN]) + tf.tile(tf.reshape(tf.gather_nd(n_sim_w, simIdx_n), [batchSize, 1, nN]), [1, nM, 1])')
        },
    '4': {
        'var': '^(._a)|(._b)|c:',
        'np': ('(m_b * m_sim ** m_a).sum(axis=0) + np.eye(nM)',
           '(n_b * n_sim ** n_a).sum(axis=0) + np.eye(nN)',
           'np.broadcast_to(m_sim_w[m, :].reshape((nM, 1)), (nM, nN)) + np.broadcast_to(n_sim_w[n, :].reshape((1, nN)), (nM, nN))'),
        'tf': ('tf.reduce_sum(tf.tile(m_b, [batchSize, 1, nM, nM]) * m_sim ** tf.tile(m_a, [batchSize, 1, nM, nM]), axis=1) + eyeM_batch',
           'tf.reduce_sum(tf.tile(n_b, [batchSize, 1, nN, nN]) * n_sim ** tf.tile(n_a, [batchSize, 1, nN, nN]), axis=1) + eyeN_batch',
           'tf.tile(tf.reshape(tf.gather_nd(m_sim_w, simIdx_m), [batchSize, nM, 1]), [1, 1, nN]) + tf.tile(tf.reshape(tf.gather_nd(n_sim_w, simIdx_n), [batchSize, 1, nN]), [1, nM, 1])')
        },
    '4n': {
        'var': '^(._a)|(._b)|c:',
        'np': ('(((m_sim >= 0) * 2 - 1) * m_b * np.absolute(m_sim) ** m_a).sum(axis=0) + np.eye(nM)',
           '(((n_sim >= 0) * 2 - 1) * n_b * np.absolute(n_sim) ** n_a).sum(axis=0) + np.eye(nN)',
           'np.broadcast_to(m_sim_w[m, :].reshape((nM, 1)), (nM, nN)) + np.broadcast_to(n_sim_w[n, :].reshape((1, nN)), (nM, nN))'),
        'tf': ('tf.reduce_sum((tf.cast(m_sim >= 0, tf.float32) * 2 - 1) * tf.tile(m_b, [batchSize, 1, nM, nM]) * tf.abs(m_sim) ** tf.tile(m_a, [batchSize, 1, nM, nM]), axis=1) + eyeM_batch',
           'tf.reduce_sum((tf.cast(n_sim >= 0, tf.float32) * 2 - 1) * tf.tile(n_b, [batchSize, 1, nN, nN]) * tf.abs(n_sim) ** tf.tile(n_a, [batchSize, 1, nN, nN]), axis=1) + eyeN_batch',
           'tf.tile(tf.reshape(tf.gather_nd(m_sim_w, simIdx_m), [batchSize, nM, 1]), [1, 1, nN]) + tf.tile(tf.reshape(tf.gather_nd(n_sim_w, simIdx_n), [batchSize, 1, nN]), [1, nM, 1])')
        }
}




'''
------------------------------------------------------------
Component functions
------------------------------------------------------------
'''
#Initialize distance
def gen_ini_dist(m_dists, n_dists, _cf):

    #Deal with empty
    #Create distance matrix, diagonal = 0, others = 2
    if len(m_dists) == 0 and _cf == False: m_dists = [-(np.eye(nM) * 2) + 2]
    if len(n_dists) == 0: n_dists = [-(np.eye(nN) * 2) + 2]

    #Transform input into proper format
    m_dists = np.stack(m_dists) if len(m_dists) > 0 else np.empty((0, 0)) #Deal with CF only
    n_dists = np.stack(n_dists)

    return m_dists, n_dists

#Initialize weight
def gen_ini_w(m_a, n_a, m_b, n_b):

    #Deal with empty
    if len(m_a) == 0: m_a = [1]; m_b = [1]
    if len(n_a) == 0: n_a = [1]; n_b = [1]

    #Transform input into proper format
    m_a = np.array(m_a).reshape((len(m_a), 1, 1))
    n_a = np.array(n_a).reshape((len(n_a), 1, 1))
    m_b = np.array(m_b).reshape((len(m_b), 1, 1))
    n_b = np.array(n_b).reshape((len(n_b), 1, 1))

    return m_a, n_a, m_b, n_b

#Prepare pref_train for a particular target cell as rating reference
#Prepare the truth adjusted by the col and row effects
#Prepare a mask masking target self and all nan cells
def gen_pref8mask(m, n, _colMask):

    #--Pref
    #Mask the pref_nan to acquire training (reference) data
    pref_train = pref_nan.copy()
    truth = pref_train[m, n]
    pref_train[m, n] = np.nan
    
    #Mask the entire column (simulate a new product which has no rating)
    if _colMask: pref_train[:, n] = np.nan
    
    #Adjust the true preference by the col and row effects
    mMean, nMean = getMean(pref_train)
    truth -= mMean[m] + nMean[n]

    #Impute nan with total mean and adjust by column and row effects (demean)
    pref_train = imputation(pref_train)

    #--Mask
    #Remove self from the matrix 
    isnan_inv_mn = isnan_inv.copy()
    isnan_inv_mn[m, n] = False

    #Remove the entire column from the matrix
    if _colMask: isnan_inv_mn[:, n] = False

    return pref_train, isnan_inv_mn, truth

#Compute CF (if needed), transform distance to similarity
def gen_dist2sim(m_dists, n_dists, _cf, pref_train, _negSim):

    #If we are going to include CF dist
    #Each rating uses the corresponding cf similarity
    if _cf:
        #Get CF dist and append to m_dists
        #nf = number of features used in SVD
        m_dist_cf = SVD(pref_train, nf=20)
        m_dist_cf = m_dist_cf.reshape((1, ) + m_dist_cf.shape)
        m_dists_processed = np.concatenate((m_dist_cf, m_dists), axis=0) if len(m_dists) > 0 else np.stack(m_dist_cf)
    
    else: m_dists_processed = m_dists

    #Flip distances (2 to 0) to similarities (0 to 1 or -1 to 1)
    #https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/generated/scipy.spatial.distance.cosine.html
    m_sim = 1 - m_dists_processed / 2 ** (not _negSim)
    n_sim = 1 - n_dists / 2 ** (not _negSim)

    return m_sim, n_sim




'''
------------------------------------------------------------
Ensemble model weight learning
------------------------------------------------------------
'''
#--Prepare data
#Initialize data
def gen_iniData(m_dists, n_dists, _cf):
    m_dists, n_dists = gen_ini_dist(m_dists, n_dists, _cf)
    return m_dists, n_dists, len(m_dists) + _cf, len(n_dists)

#Compile input dataset
def gen_npDataset(m_dists, n_dists, _cf, _negSim, _colMask):

    #Empty containers
    pref_trains, masks, m_sims, n_sims, truths, ms, ns = ([] for i in range(7))

    #Loop for each example
    for m in range(nM):
        for n in gameRatedByRater[m]:

            #Prepare all required inputs
            pref_train, mask, truth = gen_pref8mask(m, n, _colMask)
            m_sim, n_sim = gen_dist2sim(m_dists, n_dists, _cf, pref_train, _negSim)

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
def gen_learnWeight(exp, title, m_dists, n_dists, _cf, nRef, nEpoch, globalStep=0, lRate=0.5, batchSize=-1, _negSim=None, _colMask=False, _shuffle=False, _graph=False):

    #--Log
    title += ' (${}, nRef={}, lRate={}, bSize={}) ({})'.format(exp, nRef, lRate, batchSize, _currentData)
    logger.info('-' * 60)
    logger.info(title)


    #--Initialization
    #Update options
    if _negSim == None and re.search('n$', exp): _negSim = True

    #Prepare raw data
    m_dists_processed, n_dists_processed, nMDist, nNDist = gen_iniData(m_dists, n_dists, _cf)

    if DEBUG: print(m_dists_processed)
    if DEBUG: print(n_dists_processed)
    if DEBUG: print(nMDist)
    if DEBUG: print(nNDist)

    dataset_np, nExample = gen_npDataset(m_dists_processed, n_dists_processed, _cf, _negSim, _colMask)

    if DEBUG: print(dataset_np['m_sims'][0])
    if DEBUG: print(dataset_np['n_sims'][0])
    
    if batchSize == -1: batchSize = nExample #An epoch as a batch
    eyeM_batch = np.broadcast_to(np.eye(nM).reshape(1, nM, nM), (batchSize, nM, nM))
    eyeN_batch = np.broadcast_to(np.eye(nN).reshape(1, nN, nN), (batchSize, nN, nN))

    #Reset graph
    tf.reset_default_graph()


    #--Variables to be learnt
    #With an additional dimension 0 to accommodate batch
    m_a = tf.Variable(np.ones([1, nMDist, 1, 1]), name='m_a', dtype=tf.float32)
    n_a = tf.Variable(np.ones([1, nNDist, 1, 1]), name='n_a', dtype=tf.float32)
    m_b = tf.Variable(np.ones([1, nMDist, 1, 1]), name='m_b', dtype=tf.float32)
    n_b = tf.Variable(np.ones([1, nNDist, 1, 1]), name='n_b', dtype=tf.float32)
    c = tf.Variable(0.0, name='c', dtype=tf.float32)


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
    dataset = dataset.repeat() #Repeat indefinitely
    if _shuffle: dataset = dataset.shuffle(buffer_size=nExample, seed=1)
    dataset = dataset.batch(batchSize)
    iterator = dataset.make_initializable_iterator()
    pref_train, mask, m_sim, n_sim, pref_true, m_id, n_id = iterator.get_next()


    #--Operations
    #Intermediate
    simIdx_m = tf.stack([np.arange(batchSize), m_id], axis=1)
    simIdx_n = tf.stack([np.arange(batchSize), n_id], axis=1)
    M_SIM_W, N_SIM_W, MN_SIM = EXP[exp]['tf']
    m_sim_w = eval(M_SIM_W)
    n_sim_w = eval(N_SIM_W)
    mn_sim = eval(MN_SIM)
    mn_sim_mask = tf.reshape(tf.boolean_mask(mn_sim, mask), [batchSize, -1])
    pref_train_mask = tf.reshape(tf.boolean_mask(pref_train, mask), [batchSize, -1])

    #Prediction
    if nRef == -1:
        pred = c + tf.reduce_sum(pref_train_mask * mn_sim_mask, axis=1) / (tf.reduce_sum(mn_sim_mask, axis=1) + 1e-10)
    else:
        _, refIdx = tf.nn.top_k(mn_sim_mask, k=nRef, sorted=True)
        refIdx = refIdx[:, :nRef]
        ax = np.broadcast_to(np.arange(batchSize).reshape((batchSize, 1)), (batchSize, nRef))
        refIdx = tf.stack([ax, refIdx], axis=2)
        
        pred = c + tf.reduce_sum(tf.gather_nd(pref_train_mask * mn_sim_mask, refIdx), axis=1) / (tf.reduce_sum(tf.gather_nd(mn_sim_mask, refIdx), axis=1) + 1e-10)
    
    #Cost (SE)
    cost = tf.reduce_sum((pred - pref_true) ** 2)


    #--Optimizer, initializer, saver
    #Select only part of the vars to be trained
    trainVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, EXP[exp]['var'])
    opt = tf.train.AdamOptimizer(lRate).minimize(cost, var_list=trainVars)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=5)


    #--Session
    #For cost graphing
    costs = []

    with tf.Session() as sess:

        #Initialize vars/restore from checkpoint
        if globalStep == 0: sess.run(init)
        else: saver.restore(sess, './../data/checkpoint/emb-update-{}'.format(globalStep))

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

        if DEBUG: recordP = []

        #Loop over number of epochs
        for ep in range(nEpoch):

            #Get track of the cost of each epoch
            cost_epoch = 0

            #Loop over number of example
            for _ in range(math.ceil(nExample / batchSize)):

                #Run operation
                _, cost_batch, p = sess.run([opt, cost, pred])

                if DEBUG and ep == nEpoch - 1: mt, nt, mnt = sess.run([m_sim_w, n_sim_w, mn_sim])
                if DEBUG and ep == nEpoch - 1: recordP.append(p)

                #Tally the cost
                cost_epoch += cost_batch

            if ep % 1 == 0: #For graphing
                costs.append(cost_epoch)

            if ep >= 5: #For convergence termination
                if sum(abs(np.array(costs[-4:-1]) - np.array(costs[-3:]))) <= 0.3:
                    logger.info('Cost after epoch %i: %f' % (ep, cost_epoch))
                    break

            if ep % 10 == 0: #For logging cost progress
                logger.debug('Cost after epoch %i: %f' % (ep, cost_epoch))
            
            if ep + 1 == nEpoch: #Dealing with early termination
                logger.info('Failed to converge. Terminated with cost after {} epochs: {}'.format(nEpoch, cost_epoch))

        #Graphing the change of the costs
        if _graph:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per epoch)')
            plt.title("Learning rate = {}, batch size = {}".format(lRate, batchSize))
            plt.show()
            plt.close()

        #Output
        output = sess.run({'m_a': m_a.reshape(-1), 'n_a': n_a.reshape(-1), 'm_b': m_b.reshape(-1), 'n_b': n_b.reshape(-1), 'c': [c]})
        logger.info('m_a: ' + str(list(output['m_a'])))
        logger.info('n_a: ' + str(list(output['n_a'])))
        logger.info('m_b: ' + str(list(output['m_b'])))
        logger.info('n_b: ' + str(list(output['n_b'])))
        logger.info('c: ' + str(list(output['c'])))

        # saver.save(sess, './../data/checkpoint/gen_weight_{}'.format(title), global_step=globalStep + nEpoch)

        #Format for plugging into the model function
        output['exp'] = exp
        output['m_dists'] = m_dists
        output['n_dists'] = n_dists
        output['_cf'] = _cf
        output['nRef'] = nRef
        output['title'] = title
        output['_colMask'] = _colMask
        output['_negSim'] = _negSim

        if DEBUG: print('m_sim_w', mt[0])
        if DEBUG: print('n_sim_w', nt[0])
        if DEBUG: print('mn_sim', mnt[0])
        if DEBUG: print('truths', dataset_np['truths'])
        if DEBUG: print('predictions at {} epoch'.format(nEpoch), recordP)

        return output


DEBUG = False
#--Training and pipeline evaluate
#u_dist_person  u_dist_sat  u_dist_demo  dist_triplet  dist_review  dist_genre
output_1 = gen_learnWeight(exp='1', m_dists=[u_dist_person, u_dist_sat, u_dist_demo], n_dists=[dist_triplet, dist_review, dist_genre], _cf=True, nRef=-1, nEpoch=200, lRate=0.01, batchSize=-1, title='All')
predictions_1, metrics_1 = gen_model(**output_1)
output_2 = gen_learnWeight(exp='1', m_dists=[], n_dists=[], _cf=True, nRef=-1, nEpoch=100, lRate=0.01, batchSize=-1, title='CF')
predictions_2, metrics_2 = gen_model(**output_2)

output_base0 = gen_learnWeight(exp='1', m_dists=[], n_dists=[], _cf=False, nRef=-1, nEpoch=100, lRate=1, batchSize=-1, title='Base_none')
predictions_base0, metrics_base0 = gen_model(**output_base0)
output_base1 = gen_learnWeight(exp='1', m_dists=[np.eye((nM))], n_dists=[], _cf=False, nRef=-1, nEpoch=100, lRate=0.1, batchSize=-1, title='Base_eyeM')
predictions_base1, metrics_base1 = gen_model(**output_base1)
output_base2 = gen_learnWeight(exp='1', m_dists=[], n_dists=[np.eye((nN))], _cf=False, nRef=-1, nEpoch=100, lRate=0.1, batchSize=-1, title='Base_eyeN')
predictions_base2, metrics_base2 = gen_model(**output_base2)




'''
------------------------------------------------------------
Model

- Read in the data and functions in 1. and 2. by hand.
- Note the default similarity is np.eyes not np.ones.
  Therefore, different from setting an arbitrary sim matrix and weight 0.
------------------------------------------------------------
'''
#--Average similarity
def gen_model(exp, nRef, m_dists, n_dists, _cf, m_a, n_a, m_b, n_b, c, title, _negSim=False, _colMask=False, graph=False):
    
    #Initialize
    m_dists, n_dists = gen_ini_dist(m_dists, n_dists, _cf)
    m_a, n_a, m_b, n_b = gen_ini_w(m_a, n_a, m_b, n_b)

    #Prepare empty truth and prediction hulls
    truths_nan = np.full(shape=pref_nan.shape, fill_value=np.nan)
    predictions_nan = np.full(shape=pref_nan.shape, fill_value=np.nan)

    #Prediction, each cell by each cell
    for m in range(nM):
        for n in gameRatedByRater[m]:

            if DEBUG:
                if m != 2 or n != 1: continue
            
            #Prepare the reference ratings
            #Prepare the mask remove self and nan from the matrix 
            pref_train, mask, truths_nan[m, n] = gen_pref8mask(m, n, _colMask=_colMask)

            if DEBUG: print('pref_train', pref_train)
            if DEBUG: print('mask', mask)

            #Compute CF, transform distance to similarity
            m_sim, n_sim = gen_dist2sim(m_dists, n_dists, _cf, pref_train, _negSim)

            #Unpack model expression
            M_SIM_W, N_SIM_W, MN_SIM = EXP[exp]['np']
            
            #Combine weighting and similarity
            m_sim_w = eval(M_SIM_W)
            n_sim_w = eval(N_SIM_W)

            if DEBUG: print('m_sim_w', m_sim_w)
            if DEBUG: print('n_sim_w', n_sim_w)

            #Combine two types of similarities
            mn_sim = eval(MN_SIM)

            if DEBUG: print('mn_sim', mn_sim)

            #Use negative sign to reverse sort
            #Acquire only nRef references
            #(index is in flatten and nan removed)
            refIdx = np.argsort(-mn_sim[mask])[:nRef]

            #Flatten, nan removed, and make prediction based on the combined similarity
            #Clipping the value to avoid 0 division
            predictions_nan[m, n] = c[0] + np.sum((pref_train[mask] * mn_sim[mask])[refIdx]) / (np.sum(mn_sim[mask][refIdx]) + 1e-10)

            if DEBUG: print('ref_rating', pref_train[mask][refIdx])
            if DEBUG: print('ref_sim', mn_sim[mask][refIdx])
            if DEBUG: print('prediction', predictions_nan[m, n])
            if DEBUG: print(m, n)
            if DEBUG: return ["", ""]

    #Take non-nan entries and makes into long-form by [isnan_inv] slicing
    truths_gen = truths_nan[isnan_inv]
    predictions_gen = predictions_nan[isnan_inv]

    #Evaluation
    metrics_gen = evalModel(predictions_gen, truths_gen, nMN, title=title, graph=graph, logger=logger.info)

    #Return the predicted value
    return predictions_gen, metrics_gen


DEBUG = False
#--Operations
#Use nRef = -1 to employ all cells other than self
#u_dist_person  u_dist_demo  u_dist_sat  dist_triplet  dist_review  dist_genre
predictions_gen, metrics_gen = gen_model(exp='2', nRef=-1, m_dists=[np.ones((nM, nM))], n_dists=[np.ones((nN, nN))], _cf=False, m_a=[1], n_a=[1], m_b=[1], n_b=[1], c=[0], title='General model 1')
predictions_gen, metrics_gen = gen_model(exp='1', nRef=-1, m_dists=[], n_dists=[], _cf=False, m_a=[1], n_a=[], m_b=[1], n_b=[], c=[0], title='General model 2')




'''
------------------------------------------------------------
K-fold CV, comparing models
------------------------------------------------------------
'''
#--Initialization
#Acquire k-fold ids
K_FOLD = 2
id_train, id_test = kFold(K_FOLD, nMN_whole, seed=1)


#--Provide learning parameters
#Common parameters
gen_learnWeight_kFold = partial(gen_learnWeight, nRef=-1, nEpoch=30, lRate=0.1, batchSize=-1)

#Parameters to loop over
#u_dist_person  u_dist_demo  u_dist_sat  dist_triplet  dist_review  dist_genre
paras = {
'exp': ['1'],
# 'exp': ['1', '2', '2n', '3', '3n', '4', '4n'],
'para_key': ['title', 'm_dists', 'n_dists', '_cf'],
'para': [
    ['Review', [], [dist_review], False],
    # ['Sat', [u_dist_sat], [], False],
    # ['Person', [u_dist_person], [], False],
    # ['CF', [], [], True],
    ['CF+review', [], [dist_review], True],
    # ['CF+sat', [u_dist_sat], [], True],
    # ['CF+person', [u_dist_person], [], True],
    ['CF+sat+person+review', [u_dist_sat, u_dist_person], [dist_review], True]
]
}


#--Learn weights and evaluate with each fold
for exp in paras['exp']:

    #Log title
    logger.info('=' * 60)
    logger.info('Expression ' + exp)

    for para in paras['para']:
        #Record metrics of a particular para combination
        kMse, kCor, kRho = ([] for i in range(3))
        
        for i in range(K_FOLD):
            #Using training set to learn the weights
            gen_preprocessing_kFold(i + 1, 'training')
            para_dic = dict.fromkeys(paras['para_key'])
            para_dic.update(zip(paras['para_key'], para))
            output = gen_learnWeight_kFold(exp=exp, **para_dic)

            #Using test set to observe the performance
            gen_preprocessing_kFold(i + 1, 'test')
            _, metrics = gen_model(**output)

            kMse.append(metrics[0])
            kCor.append(metrics[1])
            kRho.append(metrics[2])

        #Log the performance of each para combination by averaging performance of all folds 
        logger.info('-' * 60)
        logger.info('-' * 60)
        logger.info(output['title'])
        logger.info('Average MSE = {}'.format(np.mean(kMse)))
        logger.info('Average correlation = {}'.format(np.mean(kCor)))
        logger.info('Average rankCorrelation = {}'.format(np.mean(kRho)))
        logger.info('-' * 60)
