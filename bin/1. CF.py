import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from util import *


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


#--Personality distance
person = np.genfromtxt(r'../data/personality_satisfaction.csv', delimiter=',', skip_header=1)
u_dist_person = squareform(pdist(person[:, :5], 'cosine')) #0:4 = personality; 5:7 = satisfaction


#--Demographic distance
#Get survey data
survey = pd.read_csv(r'../data/raw/survey.csv').drop_duplicates(subset='respondent')

#Sort the survey data to sync preference data order
p_order = pd.read_csv(r'../data/respondent-id.txt', names=['respondent'])
survey = p_order.merge(survey, on='respondent')

#Expand categorical data and normalize
demo = pd.get_dummies(survey, columns=['race']).iloc[:, -9:].apply(lambda col: (col - col.mean()) / col.std())

#Compute dist
u_dist_demo = squareform(pdist(demo, 'cosine'))




'''
------------------------------------------------------------
Component functions
------------------------------------------------------------
'''
#--nan imputation by column mean
#Return imputed matrix, column mean
def imputation(matrix):
    
    #Compute column and row effect
    _, nMean, mMean = deMean(matrix)

    #Find nan iloc
    naniloc = np.where(np.isnan(matrix))

    #Insert appropriate value into the matrix where is nan
    #np.take is faster than fancy indexing i.e. nMean[[1, 3, 5]]
    matrix[naniloc] = np.nanmean(matrix) + np.take(nMean, naniloc[1]) + np.take(mMean, naniloc[0])

    #Substract mean, col and row effects from pref
    matrix -= (np.reshape(nMean, (1, len(nMean))) + np.reshape(mMean, (len(mMean), 1)))

    return matrix


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
def reference_byRater(dist_target, pref_nan, pref_train, n, nRef, ifRand):

    #Sort the rater by distance and remove self
    reference_rater = np.delete(np.argsort(dist_target), 0)

    #If in random experiment, randomize order
    if ifRand == True:
        tmp = reference_rater.tolist()
        reference_rater = random.sample(tmp, k=len(tmp))

    #Make reference rating and distance
    reference_rating = []
    reference_dist = []
    for rater in reference_rater:

        #Skip nan
        if np.isnan(pref_nan[rater, n]): continue
        
        #Acquire only nRef references
        if len(reference_rating) == nRef: break

        reference_rating.append(pref_train[rater, n])
        reference_dist.append(dist_target[rater])

    return reference_rating, reference_dist


#--CF prediction of the left out
#m, n specify the left out rating
#mode changes the way prediction is computed
#Return predicted score
def CF(pref_nan, u_dist, m, n, nRef, mode, ifRand):

    #Mask the pref_nan to acquire the training data
    pref_train = pref_nan.copy()
    pref_train[m, n] = np.nan

    #Impute nan with total mean and adjust by column and row effects
    pref_train = imputation(pref_train)

    #Perform SVD for user distance matrix
    #Or use precomputed distance matrix
    if u_dist is None: u_dist = SVD(pref_train, nf=20)

    #Sort, remove self, and find the best matched raters and their ratings
    reference_rating, reference_dist = reference_byRater(u_dist[m, :], pref_nan, pref_train, n, nRef, ifRand)

    #Prediction
    #Remove column and row effects
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


#--Leave-one-out CF implementation
def implementation_cf(nRef, ifRand=False, graph=False):

    #Parameters
    nRef, mode = (nRef, '0')

    #Prediction
    predictions = recLoo(recFunc=CF, dist=None, nRef=nRef, mode=mode, ifRand=ifRand)

    #Evaluation
    mse = np.sum(np.square(predictions - prefs) / nMN)
    cor = np.corrcoef(predictions, prefs)
    print('-' * 60)
    print('CF mode {} (reference = {})'.format(mode, nRef))
    print('MSE =', mse)
    print('Correlation =', cor[0, 1])

    #Graphing
    if graph: scatter([prefs, predictions], ['prefs', 'predictions'])

    #Return the predicted value
    return predictions, cor[0, 1]

#Implement
predictions, _ = implementation_cf(10, graph=True)

#Implement with different numbers of reference
multiImplement(np.arange(1, 81), implementation_cf, nRand=30, titleLabel='Cf')


#--Personality implementation
def implementation_person(nRef, ifRand=False, graph=False):

    #Parameters
    nRef_person, mode_person = (nRef, '0')

    #Prediction
    predictions_person = recLoo(recFunc=CF, dist=u_dist_person, nRef=nRef_person, mode=mode_person, ifRand=ifRand)

    #Evaluation
    mse_person = np.sum(np.square(predictions_person - prefs) / nMN)
    cor_person = np.corrcoef(predictions_person, prefs)
    print('-' * 60)
    print('Personality mode {} (reference = {})'.format(mode_person, nRef_person))
    print('MSE =', mse_person)
    print('Correlation =', cor_person[0, 1])

    #Graphing
    if graph: scatter([prefs, predictions_person], ['prefs', 'predictions_person'])

    #Return the predicted value
    return predictions_person, cor_person[0, 1]

#Implement
predictions_person, _ = implementation_person(10, graph=True)

#Implement with different numbers of reference
multiImplement(np.arange(1, 81), implementation_person, nRand=30, titleLabel='Person')


#--Demographic implementation
def implementation_demo(nRef, ifRand=False, graph=False):

    #Parameters
    nRef_demo, mode_demo = (nRef, '0')

    #Prediction
    predictions_demo = recLoo(recFunc=CF, dist=u_dist_demo, nRef=nRef_demo, mode=mode_demo, ifRand=ifRand)

    #Evaluation
    mse_demo = np.sum(np.square(predictions_demo - prefs) / nMN)
    cor_demo = np.corrcoef(predictions_demo, prefs)
    print('-' * 60)
    print('Demographic mode {} (reference = {})'.format(mode_demo, nRef_demo))
    print('MSE =', mse_demo)
    print('Correlation =', cor_demo[0, 1])

    #Graphing
    if graph: scatter([prefs, predictions_demo], ['prefs', 'predictions_demo'])

    #Return the predicted value
    return predictions_demo, cor_demo[0, 1]

#Implement
predictions_demo, _ = implementation_demo(10, graph=True)

#Implement with different numbers of reference
multiImplement(np.arange(1, 81), implementation_demo, nRand=30, titleLabel='demo')




'''
------------------------------------------------------------
Experimental
------------------------------------------------------------
'''
#--CF by optimization
#Too many parameters, perhaps can use some regularization
tf.reset_default_graph()
learning_rate = 0.01

#Compute 2 matrices, whose multiplication approximates the original user-restaurant matrix
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
