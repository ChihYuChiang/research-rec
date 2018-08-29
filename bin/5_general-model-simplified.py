from util import *


'''
------------------------------------------------------------
Component functions

- Replace the original functions in 4.
------------------------------------------------------------
'''
def gen_preprocessing_kFold(foldId):

    #Fold id 1 -> 0. Fold ID starts from 1.
    foldId -= 1

    #Load data and create dummies
    data_raw = pd.read_csv(r'../data/DT_2_long.csv')[['respondent', 'core_id', 'preference']]
    data_raw_X = data_raw.iloc[id_X[foldId], :]
    data_raw_Y = data_raw.iloc[id_Y[foldId], :]
    data_dummy = pd.get_dummies(data_raw, columns=['respondent', 'core_id'])
    data_dummy_X = data_dummy.iloc[id_X[foldId], :]
    data_dummy_Y = data_dummy.iloc[id_Y[foldId], :]

    #Use X in linear model for demean
    lm = LinearRegression()
    lm.fit(data_dummy_X.iloc[:, 1:], data_dummy_X[['preference']])

    #Compute demeaned data
    #5.5 as a constant to make the value positive (dealing with the scikitlearn bug)
    data_raw_X['demean'] = data_dummy_X[['preference']] - lm.predict(data_dummy_X.iloc[:, 1:]) + 5.5
    data_raw_Y['demean'] = data_dummy_Y[['preference']] - lm.predict(data_dummy_Y.iloc[:, 1:]) + 5.5

    #Reset and initialize
    data_X, data_Y = DataContainer(), DataContainer()
    data_X.pref_nan = pd.DataFrame(index=pd.read_csv(r'../data/raw/raw_preference.csv').ResponseId.unique(), columns=range(1, 51))
    data_Y.pref_nan = pd.DataFrame(index=pd.read_csv(r'../data/raw/raw_preference.csv').ResponseId.unique(), columns=range(1, 51))

    #Produce item-rater matrix
    for _, r in data_raw_X.iterrows():
        data_X.pref_nan.set_value(r.respondent, r.core_id, r.demean)
    data_X.pref_nan = data_X.pref_nan.values.astype(np.float32)

    for _, r in data_raw_Y.iterrows():
        data_Y.pref_nan.set_value(r.respondent, r.core_id, r.demean)
    data_Y.pref_nan = data_Y.pref_nan.values.astype(np.float32)

    #Update data by pref_nan
    data_X.updateByNan()
    data_Y.updateByNan()
    
    #Log
    markers.CURRENT_DATA = '#{}/{}'.format(foldId + 1, options.K_FOLD)

    return data_X, data_Y

#Get data directly from globals
#For reusing the structure in 4.
def gen_pref8mask(pref_train, isnan_inv_mn, m, n, _colMask):

    #--Pref
    #Use Y to get the truth
    truth = data_Y.pref_nan[m, n]    

    #--Mask
    #Use X to replace the pref_train from Y
    pref_train = data_X.pref_nan.copy()
    isnan_inv_mn = data_X.isnan_inv.copy()

    #Mask the entire column (simulate a new product which has no rating)
    if _colMask:
        pref_train[:, n] = np.nan
        isnan_inv_mn[:, n] = False

    return pref_train, isnan_inv_mn, truth
    



'''
------------------------------------------------------------
Model
------------------------------------------------------------
'''
#--Compute CF using X
u_dist_cf = SVD(imputation(data_X.pref_nan, imValue=np.nanmean(data_X.pref_nan)))


#--Prepare pref_train, mask, and the truth
#90% X, 10% Y
#Avoid autocorrelation when using all - 1 samples predicts a target
id_X, id_Y = kFold(3, data_whole.nMN, seed=1)

#Acquire X and Y
foldId = 1
data_X, data_Y = gen_preprocessing_kFold(foldId)
data_Y.listData()


#--Predicting Y by X with sim combinations
#Inherit
#u_dist_person  u_dist_sat  u_dist_demo  u_dist_cf  dist_triplet  dist_review  dist_genre
output = gen_learnWeight(data=data_Y, exp='1', m_dists=[np.ones((data_Y.nM, data_Y.nM)) + np.eye(data_Y.nM)], n_dists=[np.ones((data_Y.nN, data_Y.nN)) + np.eye(data_Y.nN)], _cf=False, nRef=-1, nEpoch=50, lRate=0.01, batchSize=-1, title='All')
predictions, metrics = gen_model(**output)
output = gen_learnWeight(data=data_Y, exp='1', m_dists=[u_dist_cf], n_dists=[np.ones((data_Y.nN, data_Y.nN))], _cf=False, nRef=-1, nEpoch=50, lRate=0.01, batchSize=-1, title='All')
predictions, metrics = gen_model(**output)
output = gen_learnWeight(data=data_Y, exp='1', m_dists=[np.ones((data_Y.nM, data_Y.nM)) + np.eye(data_Y.nM)], n_dists=[dist_review + np.eye(data_Y.nN) * 2], _cf=False, nRef=-1, nEpoch=50, lRate=0.01, batchSize=-1, title='All')
predictions, metrics = gen_model(**output)
output = gen_learnWeight(data=data_Y, exp='1', m_dists=[u_dist_sat + np.eye(data_Y.nM) * 2], n_dists=[dist_review + np.eye(data_Y.nN) * 2], _cf=False, nRef=-1, nEpoch=50, lRate=0.01, batchSize=-1, title='All')
predictions, metrics = gen_model(**output)

#Manual
options.DEBUG = False
options.DEBUG = True
predictions, metrics = gen_model(data=data_Y, exp='1', nRef=-1, m_dists=[u_dist_cf], n_dists=[np.ones((data_Y.nN, data_Y.nN))], _cf=False, m_a=[1], n_a=[1], m_b=[1], n_b=[1], c=[0], title='General model 2', target=[5, 14])
predictions, metrics = gen_model(data=data_Y, exp='1', nRef=-1, m_dists=[np.ones((data_Y.nM, data_Y.nM))], n_dists=[dist_triplet], _cf=False, m_a=[1], n_a=[1], m_b=[1], n_b=[1], c=[0], title='General model 2', target=[0, 3])
predictions, metrics = gen_model(data=data_Y, exp='1', nRef=-1, m_dists=[u_dist_sat], n_dists=[dist_review], _cf=False, m_a=[1], n_a=[1], m_b=[1], n_b=[1], c=[0], title='General model 2')

pred = np.array([1, 2, 3])
pref_true = np.array([2, 2, 3])
batchSize = 2
-np.abs((batchSize * np.sum(pred * pref_true) - np.sum(pred) * np.sum(pref_true)) / (((batchSize * np.sum(pred ** 2) - np.sum(pred) ** 2) ** 0.5) * ((batchSize * np.sum(pref_true ** 2) - np.sum(pref_true) ** 2) ** 0.5)))



'''
------------------------------------------------------------
Comparing models
------------------------------------------------------------
'''
#--Provide learning parameters
#Common parameters
gen_learnWeight_kFold = partial(gen_learnWeight, nRef=-1, nEpoch=1000, lRate=0.1, batchSize=-1)

#Parameters to loop over
#u_dist_person  u_dist_sat  u_dist_demo  u_dist_cf  dist_triplet  dist_review  dist_genre
paras = {
# 'exp': ['1', '2', '2n', '3', '3n', '4', '4n'],
'exp': ['1', '2', '3', '4'],
'para_key': ['title', 'm_dists', 'n_dists'],
'para': [
    ['EyeM', [], [np.ones((nN, nN))]],
    ['EyeN', [np.ones((nM, nM))], []],
    ['Review', [], [dist_review]],
    # ['Sat', [u_dist_sat], []],
    ['Person', [u_dist_person], []],
    ['CF', [], []],
    ['CF+review', [], [dist_review]],
    # ['CF+sat', [u_dist_sat], []],
    ['CF+person', [u_dist_person], []],
    ['CF+person+review', [u_dist_person], [dist_review]]
    # ['CF+sat+person+review', [u_dist_sat, u_dist_person], [dist_review]]
]}


#--Learn weights and evaluate
para_dic = dict.fromkeys(paras['para_key'])
for exp in paras['exp']:

    #Log title
    logger.info('=' * 60)
    logger.info('Expression ' + exp)

    for para in paras['para']:
        
        #Learn the weights
        para_dic.update(zip(paras['para_key'], para))
        output = gen_learnWeight_kFold(exp=exp, **para_dic)

        #Evaluate the performance
        _, metrics = gen_model(**output)

        #Log the performance of each para combination by averaging performance of all folds 
        logger.info('-' * 60)
        logger.info('Combination Summary')
        logger.info('-' * 60)
        logger.info(output['title'])
        logger.info('MSE = {}'.format(metrics[0]))
        logger.info('Correlation = {}'.format(metrics[1]))
        logger.info('RankCorrelation = {}'.format(metrics[2]))
        logger.info('-' * 60)

        #Log csv
        #Title, MSE, cor, rho
        logger_metric.info(', '.join(['\"{}\"'.format(output['title']), str(metrics[0]), str(metrics[1]), str(metrics[2])]))