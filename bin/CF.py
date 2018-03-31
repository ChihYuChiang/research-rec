import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from util import preprocessing


'''
------------------------------------------------------------
Data
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
Component functions
------------------------------------------------------------
'''
#--nan imputation by column mean
#Return imputed matrix, column mean
def imputation(matrix):
    
    #Compute column mean
    nMean = np.nanmean(matrix, axis=0)

    #Find nan iloc
    naniloc = np.where(np.isnan(matrix))

    #Insert appropriate value into the matrix where is nan
    #np.take is faster than fancy indexing i.e. nMean[[1, 3, 5]]
    matrix[naniloc] = np.take(nMean, naniloc[1])

    return matrix, nMean


#--SVD
#Return user pair-wise distance matrix
def SVD(matrix, nf=10):

    #Complete SVD
    #Result shape: (nM, nM) (nN, ) (nN, nN)
    u, s, vh = np.linalg.svd(matrix)

    #Truncate SVD (use t largest singular values)
    u_t = u[:, 0:nf]

    #Distance matrix
    u_dist = squareform(pdist(u_t, 'cosine'))
    
    return u_dist


#--Find the best matched raters and make reference
#Return reference rating vec and corresponding distance vec
def reference(dist_target, pref_nan, n, nRef):

    #Sort the rater by distance and remove self
    reference_rater = np.delete(np.argsort(dist_target), 0)

    #Make reference rating and distance
    reference_rating = []
    reference_dist = []
    for rater in reference_rater:

        #Skip nan
        if np.isnan(pref_nan[rater, n]): continue
        
        #Acquire only nRef references
        if len(reference_rating) == nRef: break

        reference_rating.append(pref_nan[rater, n])
        reference_dist.append(dist_target[rater])

    return reference_rating, reference_dist


#--CF prediction of the left out
#m, n specify the left out rating
#mode changes the way prediction is computed
#Return predicted score
def CF(pref_nan, u_dist, m, n, nRef, mode):

    #Mask the pref_nan to acquire the training data
    pref_train = pref_nan.copy()
    pref_train[m, n] = np.nan

    #Impute nan with column mean
    pref_train, nMean = imputation(pref_train)

    #If mode 1, substract col mean from pref
    if mode == 1: pref_train -= nMean

    #Perform SVD for user distance matrix
    #Or use precomputed distance matrix
    if u_dist is None: u_dist = SVD(pref_train, nf=20)

    #Sort, remove self, and find the best matched raters and their ratings
    reference_rating, reference_dist = reference(u_dist[m, :], pref_nan, n, nRef)

    #Prediction
    #Subtract column mean to see the prediction of personal preference
    #Dist as weight -> transform back to -1 to 1
    computation = {
        0: np.mean(reference_rating) - nMean[n],
        1: np.mean(reference_rating) - nMean[n],
        2: np.dot(np.array(reference_rating) - nMean[n], -(np.array(reference_dist) - 1))
    }
    prediction = computation[mode]

    return prediction


#--Leave-one-out implementation
#Return predicted score in long-form (de-colmeaned) and CF mode
def CF_loo(u_dist, nRef, mode):

    #Operation
    predictions_nan = np.full(shape=pref_nan.shape, fill_value=np.nan)
    for m in np.arange(nM):
        for n in gameRatedByRater[m]:
            predictions_nan[m, n] = CF(pref_nan, u_dist, m, n, nRef=nRef, mode=mode)

    #Take non-nan entries and makes into long-form by [isnan_inv] slicing
    predictions = predictions_nan[isnan_inv]
    
    return predictions




'''
------------------------------------------------------------
Models
------------------------------------------------------------
'''
#--Subtract column mean for pref matrix and makes it long-form
nMean = np.broadcast_to(np.nanmean(pref_nan, axis=0), (nM, nN))
prefs = pref_nan[isnan_inv] - nMean[isnan_inv]


#--Leave-one-out CF implementation
#Parameters
nRef, mode = (10, 1)

#Prediction
predictions = CF_loo(u_dist=None, nRef=nRef, mode=mode)

#Evaluation
mse = np.sum(np.square(predictions - prefs) / nMN)
cor = np.corrcoef(predictions, prefs)
print('-' * 60)
print('CF mode {} (reference = {})'.format(mode, nRef))
print('MSE =', mse)
print('Correlation =', cor[0, 1])


#--Personality implementation
#Parameters
nRef_person, mode_person = (10, 1)

#Get user distance matrix
person = np.genfromtxt(r'../data/personality_satisfaction.csv', delimiter=',', skip_header=1)
u_dist_person = squareform(pdist(person[:, :5], 'cosine')) #0:4 = personality; 5:7 = satisfaction

#Prediction
predictions_person = CF_loo(u_dist=u_dist_person, nRef=nRef_person, mode=mode_person)

#Evaluation
mse_person = np.sum(np.square(predictions_person - prefs) / nMN)
cor_person = np.corrcoef(predictions_person, prefs)
print('-' * 60)
print('Personality mode {} (reference = {})'.format(mode_person, nRef_person))
print('MSE =', mse_person)
print('Correlation =', cor_person[0, 1])


#--Benchmark
#Column mean prediction MSE
mse_nMean = np.sum(np.square(0 - prefs) / nMN)
print('-' * 60)
print('Column mean benchmark')
print('MSE =', mse_nMean)

np.sum(np.square((predictions + predictions_person) / 2 - prefs)) / nMN
np.corrcoef(predictions + predictions_person, prefs)


#--CF and personality ensemble
tf.reset_default_graph()
learning_rate = 0.01
w = tf.Variable(0, name='weight', dtype=tf.float32)
cost = tf.reduce_sum(tf.square((w * predictions + (1 - w) * predictions_person) - prefs)) / nMN
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

costs = []
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(2000):
        _, epoch_cost = sess.run([optimizer, cost])

        if epoch % 100 == 0:
            print ('Cost after epoch %i: %f' % (epoch, epoch_cost))
        if epoch % 5 == 0:
            costs.append(epoch_cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    plt.close()

    w_trained = sess.run(w)
    print ('Trained w =', w_trained)

np.sum(np.square((w_trained * predictions + (1 - w_trained) * predictions_person) - prefs)) / nMN
np.corrcoef(w_trained * predictions + (1 - w_trained) * predictions_person, prefs)




'''
------------------------------------------------------------
Experimental
------------------------------------------------------------
'''
#--CF by optimization
#Too many parameters, perhaps can use some regularization
tf.reset_default_graph()
learning_rate = 0.01
us = tf.get_variable('us', [215, 1], dtype=tf.float32,
  initializer=tf.random_uniform_initializer())
vh = tf.get_variable("vh", [1, 50], dtype=tf.float32,
  initializer=tf.random_uniform_initializer())

init2 = tf.global_variables_initializer()
cost2 = tf.reduce_sum(tf.square(tf.boolean_mask(tf.matmul(us, vh), isnan_inv) - prefs)) / nMN
optimizer2 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost2)

costs2 = []
with tf.Session() as sess:
    sess.run(init2)

    for epoch in range(100000):
        _, epoch_cost = sess.run([optimizer2, cost2])

        if epoch % 5000 == 0:
            print ('Cost after epoch %i: %f' % (epoch, epoch_cost))
        if epoch % 1000 == 0:
            costs2.append(epoch_cost)

    plt.plot(np.squeeze(costs2))
    plt.ylabel('cost')
    plt.xlabel('iterations (per thousands)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    plt.close()

    out1 = sess.run(cost2)
    out2 = sess.run(tf.matmul(us, vh))

np.corrcoef(out2[isnan_inv], prefs)
