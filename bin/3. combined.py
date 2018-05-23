import numpy as np
from scipy.stats import t as dis_t
import matplotlib.pyplot as plt
from util import *


'''
------------------------------------------------------------
Ensemble

- Read in the data and functions in 1. and 2. by hand.
------------------------------------------------------------
'''
#--CF and personality (and content based) ensemble
def ensemble(predictions, epoch=2000, graph=False):

    #Training
    predictionStack = np.stack(predictions)
    w_trained = ensembleWeight(predictionStack, prefs, nEpoch=epoch)
    w_formatted = [w for w in flattenList(w_trained.tolist())]

    #Prediction
    predictions_en = np.sum(w_trained * predictionStack, axis=0)

    #Evaluation
    mse_en = np.sum(np.square(predictions_en - prefs) / nMN)
    cor_en = np.corrcoef(predictions_en, prefs)
    print('-' * 60)
    print('Ensemble\nweight = {}'.format(w_formatted))
    print('MSE =', mse_en)
    print('Correlation =', cor_en[0, 1])

    #Graphing
    if graph: scatter([prefs, predictions_en], ['prefs', 'predictions_en'])

    #Return the predicted value
    return predictions_en, cor_en[0, 1], w_formatted

#Implement
predictions_en, _, __ = ensemble([implementation_cf(45)[0], implementation_person(45)[0]], epoch=20000, graph=True)


#--Implement with different numbers of reference
cors_cf, cors_person, cors_en, w_cf, w_person = [], [], [], [], []

N_REF = np.arange(1, 81)
for i in N_REF:
    predictions_cf, cor_cf = implementation_cf(i)
    cors_cf.append(cor_cf)

    predictions_person, cor_person = implementation_person(i)
    cors_person.append(cor_person)

    _, cor_en, w = ensemble([predictions_cf, predictions_person], epoch=int(round((i ** 0.5) * 5000)))
    cors_en.append(cor_en)
    w_cf.append(w[0])
    w_person.append(w[1])

#Graph for the correlations
plt.plot(N_REF, cors_cf, label='CF')
plt.plot(N_REF, cors_person, label='Personality')
plt.plot(N_REF, cors_en, label='Combined')
plt.legend(loc=(1.03, 0.6))
plt.title('Ensemble correlation by number of reference')
plt.xlabel('Number of reference')
plt.ylabel('Correlation with the real score')
plt.show()
plt.close()

#Graph for the ensemble weights
fig, ax = plt.subplots()
ax.bar(N_REF, w_cf, label='CF')
ax.bar(N_REF, w_person, bottom=w_cf, label='Personality')
ax.legend(loc=(1.03, 0.6))
ax.axhline(0.5, ls='--', color='r')
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
    predictions_en, cor_en, _ = ensemble([implementation_cf(nRef)[0], implementation_person(nRef)[0]], epoch=40000)
    cor_cfEn = np.corrcoef(predictions_en, predictions_cf)[0, 1]

    #Then use online calculator..
    #https://www.psychometrica.de/correlation.html
    print('-' * 60)
    print('Number of reference =', nRef)
    print('r12 = {}\nr13 = {}\nr23 = {}'.format(cor_cf, cor_en, cor_cfEn))

cors(30)