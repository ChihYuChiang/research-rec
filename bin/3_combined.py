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

            if epoch % 10000 == 0:
                print ('Cost after epoch %i: %f' % (epoch, epoch_cost))
            if epoch % 100 == 0:
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




'''
------------------------------------------------------------
Ensemble model

- Read in the data and functions in 1. and 2. by hand.
------------------------------------------------------------
'''
#--Average prediction
#CF, personality, and content based ensemble
def ensemble_ap(predictions, nEpoch=2000, graph=False):

    #Training
    predictionStack = np.stack(predictions)
    w_trained = ensembleWeight_ap(predictionStack, prefs, nEpoch=nEpoch)
    w_formatted = [w for w in flattenList(w_trained.tolist())]

    #Prediction
    predictions_en = np.sum(w_trained * predictionStack, axis=0)

    #Evaluation
    mse_en, cor_en, rho_en = evalModel(predictions_en, prefs, nMN, title='Ensemble (average prediction)\nweight = {}'.format(w_formatted), graph=graph)

    #Return the predicted value
    return predictions_en, cor_en, w_formatted

#Implement
predictions_en, _, __ = ensemble_ap([implementation_cf(45)[0], implementation_person(45)[0], implementation_c(6)[0]], nEpoch=20000, graph=True)


#--Implement with different numbers of reference
cors_cf, cors_person, cors_en1, cors_en2 = ([] for i in range(4))
w_cf, w_person, w_text = ([[], []] for i in range(3))

predictions_text, _ = implementation_c(6) #Text
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