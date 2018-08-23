import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from functools import partial
from util import *

DEBUG = False


'''
------------------------------------------------------------
Preference data
------------------------------------------------------------
'''
#--Preprocessing pref data
#Read from file, processing without folds
pref_nan, prefs, nM, nN, nMN, isnan_inv, gameRatedByRater = preprocessing(description=False)




'''
------------------------------------------------------------
People similarity
------------------------------------------------------------
'''
#--Personality and satisfaction distance
person = np.genfromtxt(r'../data/personality_satisfaction.csv', delimiter=',', skip_header=1)

#0:4 = personality; 5:7 = satisfaction
u_dist_person = squareform(pdist(person[:, :5], 'cosine')) 
u_dist_sat = squareform(pdist(person[:, 5:], 'cosine'))


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
#--Find the best matched raters and make reference
#Return reference rating vec and corresponding distance vec
def reference_byRater(dist_target, pref_nan, pref_train, m, n, nRef, ifRand):

    #Remove self in the array
    pref_mask = pref_nan.copy()
    pref_mask[m, n] = np.nan

    #Sort the rater by distance
    reference_rater = np.argsort(dist_target)

    #If in random experiment, randomize order
    if ifRand == True:
        tmp = reference_rater.tolist()
        reference_rater = random.sample(tmp, k=len(tmp))

    #Make reference rating and distance
    reference_rating = []
    reference_dist = []
    for rater in reference_rater:

        #Skip nan
        if np.isnan(pref_mask[rater, n]): continue
        
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
    reference_rating, reference_dist = reference_byRater(u_dist[m, :], pref_nan, pref_train, m, n, nRef, ifRand)

    # print(reference_rating)
    # print(reference_dist)
    # print( 1 - np.array(reference_dist) / 2)

    #Prediction
    #Remove column and row effects
    #Dist as weight -> transform back to -1 to 1
    computation = {
        '0': np.mean(reference_rating),
        '1': np.dot(np.array(reference_rating), 1 - np.array(reference_dist) / 2)
    }
    prediction = computation[mode]

    return prediction

if DEBUG: CF(pref_nan, None, 2, 1, 10, '1', False)




if __name__ != '__main__': #Manually execute when using Jupyter

    '''
    ------------------------------------------------------------
    Models

    - Note the true pref using here gets its own demean and is supposed to be revised
    ------------------------------------------------------------
    '''
    #--Implementation wrapper
    def implementation(nRef, recFunc, dist, mode, title, ifRand=False, graph=False):
        
        #Prediction
        predictions = recLoo(recFunc=recFunc, dist=dist, nRef=nRef, mode=mode, ifRand=ifRand)

        #Create proper title for presenting result
        title = title.format(mode, nRef)

        #Evaluation
        mse, cor, rho = evalModel(predictions, prefs, nMN, title, graph=graph)

        #Return the predicted value
        return predictions, cor

    implementation_cf = partial(implementation, recFunc=CF, dist=None, mode='0', title='CF mode {} (reference = {})')
    implementation_person = partial(implementation, recFunc=CF, dist=u_dist_person, mode='0', title='Personality mode {} (reference = {})')
    implementation_demo = partial(implementation, recFunc=CF, dist=u_dist_demo, mode='0', title='Demographic mode {} (reference = {})')

    #------------------------------------------------------------

    #--CF implementation
    #Single implement
    predictions, _ = implementation_cf(10, graph=True)

    #Implement with different numbers of reference
    multiImplement(np.arange(1, 81), implementation_cf, nRand=30, titleLabel='Cf')


    #--Personality implementation
    #Single implement
    predictions_person, _ = implementation_person(10, graph=True)

    #Implement with different numbers of reference
    multiImplement(np.arange(1, 81), implementation_person, nRand=30, titleLabel='Person')


    #--Demographic implementation
    #Single implement
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
