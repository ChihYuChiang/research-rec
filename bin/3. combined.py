import numpy as np
from scipy.stats import t as dis_t
import matplotlib.pyplot as plt
from util import *


'''
------------------------------------------------------------
Component functions
------------------------------------------------------------
'''
#--Ensemble model weighting
#Average prediction
def ensembleWeight_ap(predictionStack, prefs, nEpoch=2000, graph=False):

    #Initialization
    #Minimize MSE
    tf.reset_default_graph()
    learning_rate = 0.01
    w = tf.Variable(np.zeros((len(predictionStack), 1)), name='weight', dtype=tf.float32)
    cost = tf.reduce_mean(tf.square(tf.reduce_sum(predictionStack * w, axis=0) - prefs))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    #Training
    costs = []
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(nEpoch):
            _, epoch_cost = sess.run([optimizer, cost])

            if epoch % 100 == 0:
                print ('Cost after epoch %i: %f' % (epoch, epoch_cost))
            if epoch % 5 == 0:
                costs.append(epoch_cost)

        if graph:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per fives)')
            plt.title("Learning rate = " + str(learning_rate))
            plt.show()
            plt.close()

        w_trained = sess.run(w)

    return w_trained




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

tt1 = np.array([[1, 2, 3], [4, 5, 6]], dtype='float32')
tt2 = np.array([[10, 20, 30], [60, 50, 40]], dtype='float32')
prefs = np.array([[15, 25, 35], [1.5, None, 3.5]], dtype='float32')
np.sum(np.stack([tt1, tt2]) * np.array([[[0.1]], [[0.5]]]), axis=0)
ensembleWeight_as([tt1, tt2], prefs, 2)

'''
------------------------------------------------------------
Ensemble model

- Read in the data and functions in 1. and 2. by hand.
------------------------------------------------------------
'''
"Average prediction"
#CF, personality, and content based ensemble
def ensemble_ap(predictions, nEpoch=2000, graph=False):

    #Training
    predictionStack = np.stack(predictions)
    w_trained = ensembleWeight_ap(predictionStack, prefs, nEpoch=nEpoch)
    w_formatted = [w for w in flattenList(w_trained.tolist())]

    #Prediction
    predictions_en = np.sum(w_trained * predictionStack, axis=0)

    #Evaluation
    mse_en, cor_en = evalModel(predictions_en, prefs, nMN, text_title='Ensemble (average prediction)\nweight = {}'.format(w_formatted), graph=graph)

    #Return the predicted value
    return predictions_en, cor_en, w_formatted

#Implement
predictions_en, _, __ = ensemble_ap([implementation_cf(45)[0], implementation_person(45)[0], implementation_c(1)[0]], nEpoch=20000, graph=True)


#--Implement with different numbers of reference
cors_cf, cors_person, cors_en1, cors_en2 = ([] for i in range(4))
w_cf, w_person, w_text = ([[], []] for i in range(3))

predictions_text, _ = implementation_c(1) #Text
N_REF = np.arange(1, 81)
for i in N_REF:
    predictions_cf, cor_cf = implementation_cf(i) #CF
    cors_cf.append(cor_cf)

    predictions_person, cor_person = implementation_person(i) #Personality
    cors_person.append(cor_person)

    _, cor_en1, w1 = ensemble_ap([predictions_cf, predictions_person], nEpoch=int(min(round((i ** 1.2) * 1000), 80000))) #Ensemble1
    cors_en1.append(cor_en1)

    _, cor_en2, w2 = ensemble_ap([predictions_cf, predictions_person, predictions_text], nEpoch=int(min(round((i ** 1.2) * 1500), 100000))) #Ensemble2
    cors_en2.append(cor_en2)

    w_cf[0].append(w1[0])
    w_person[0].append(w1[1])
    w_cf[1].append(w2[0])
    w_person[1].append(w2[1])
    w_text[1].append(w2[2])

#Graph for the correlations
plt.plot(N_REF, cors_cf, label='CF')
plt.plot(N_REF, cors_person, label='Personality')
plt.plot(N_REF, cors_en1, label='CF+Personality')
plt.plot(N_REF, cors_en2, label='CF+Personality+Text')
plt.legend(loc=(1.03, 0.6))
plt.title('Ensemble correlation by number of reference')
plt.xlabel('Number of reference')
plt.ylabel('Correlation with the real score')
plt.show()
plt.close()

#Graph for the ensemble weights
#ax.axhline(0.5, ls='--', color='r')
fig, ax = plt.subplots()
ax.bar(N_REF, listEWiseOp(abs, w_cf[1]), label='CF')
ax.bar(N_REF, listEWiseOp(abs, w_person[1]), bottom=listEWiseOp(abs, w_cf[1]), label='Personality')
ax.bar(N_REF, listEWiseOp(abs, w_text[1]), bottom=[sum(x) for x in zip(listEWiseOp(abs, w_cf[1]), listEWiseOp(abs, w_person[1]))], label='Text')
ax.legend(loc=(1.03, 0.6))
ax.set(xlabel='Number of reference', ylabel='Weight proportion', title='Ensemble weight proportion by number of reference')
plt.show()
plt.close()




"Average similarity"
#--Matrix prediction and evaluation
def ensemble_as(nRef, m_dists, n_dists, m_w, n_w, graph=False):
    
    #Deal with empty
    if len(m_w) == 0: m_dists = [np.eye(nM)]; m_w = [1]
    if len(n_w) == 0: n_dists = [np.eye(nN)]; n_w = [1]

    #Transform input into proper format
    m_dists = np.stack(m_dists)
    n_dists = np.stack(n_dists)
    m_w = np.array(m_w).reshape((len(m_w), 1, 1))
    n_w = np.array(n_w).reshape((len(n_w), 1, 1))

    #Save a copy of the input dists
    m_dists_input = m_dists.copy()

    #Prepare an empty prediction hull
    predictions_nan = np.full(shape=pref_nan.shape, fill_value=np.nan)

    #Prediction, each cell by each cell
    for m in range(nM):
        for n in gameRatedByRater[m]:

            #If we are going to include CF dist
            #Each rating uses the corresponding cf similarity
            if len(m_dists) < len(m_w):
                #Reset the m_dists
                m_dists = m_dists_input.copy()
                
                #Mask the pref_nan to acquire CF training data
                pref_train = pref_nan.copy()
                pref_train[m, n] = np.nan

                #Impute nan with total mean and adjust by column and row effects
                pref_train = imputation(pref_train)

                #Get CF dist and append to m_dists
                #nf = number of features used in SVD
                m_dist_cf = SVD(pref_train, nf=20)
                m_dist_cf = m_dist_cf.reshape((1, ) + m_dist_cf.shape)
                m_dists = np.concatenate((m_dists, m_dist_cf), axis=0)

            #Combine weighting and distance
            m_sim = (m_dists ** m_w).sum(axis=0)
            n_sim = (n_dists ** n_w).sum(axis=0)

            #Combine two types of distances
            mn_sim = np.matmul(m_sim[m, :].reshape((nM, 1)), n_sim[n, :].reshape((1, nN)))

            #Reference rating matrix
            pref_ref = pref_nan.copy()
            pref_ref[m, n] = np.nan

            #Acquire only nRef references

            #Make prediction based on the combined distance
            predictions_nan[m, n] = np.nansum(pref_ref * mn_sim)

    #Take non-nan entries and makes into long-form by [isnan_inv] slicing
    predictions_en = predictions_nan[isnan_inv]            

    #Evaluation
    mse_en, cor_en = evalModel(predictions_en, prefs, nMN, title='Ensemble (average similarity)', graph=graph)

    #Return the predicted value
    return predictions_en, cor_en


predictions_en, cor_en = ensemble_as(nRef=-1, m_dists=[u_dist_demo], n_dists=[], m_w=[0.5, 0.5], n_w=[])



'''
------------------------------------------------------------
Compare correlations
------------------------------------------------------------
'''
#--Independent correlations
#Fisher z transformation
#https://en.wikipedia.org/wiki/Fisher_transformation
def fisherTran(r, n):
    z = 0.5 * np.log((1 + r) / (1 - r))
    z_se = 1 / (n - 3) ** 0.5

    return (z, z_se)

#Z1 - Z2 single-tail independent t test
def studentT_ide(z1, z1_se, z2, z2_se, n):
    sp = ((z1_se ** 2 + z2_se ** 2) / 2) ** 0.5
    t = (z1 - z2) / (sp * (2 / n) ** 0.5)
    df = 2 * n - 2
    
    return 1 - dis_t.cdf(t, df)

studentT_ide(*fisherTran(0.18, 2000), *fisherTran(0.178, 2000), 2000)


#--Dependent correlations
#http://www.psychmike.com/Steiger.pdf
def cors(nRef):
    predictions_cf, cor_cf = implementation_cf(nRef)
    predictions_en, cor_en, _ = ensemble([implementation_cf(nRef)[0], implementation_person(nRef)[0]], nEpoch=40000)
    cor_cfEn = np.corrcoef(predictions_en, predictions_cf)[0, 1]

    #Then use online calculator..
    #https://www.psychometrica.de/correlation.html
    print('-' * 60)
    print('Number of reference =', nRef)
    print('r12 = {}\nr13 = {}\nr23 = {}'.format(cor_cf, cor_en, cor_cfEn))

cors(30)