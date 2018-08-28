from util import *


'''
------------------------------------------------------------
Component functions

- Replace the original functions in 4.
------------------------------------------------------------
'''
def gen_preprocessing_kFold(data_whole, data_current, foldId, _marker):
    assert _marker in ['X', 'Y'], 'Wrong marker. Use X or Y.'

    #Fold id 1 -> 0. Fold ID starts from 1.
    foldId -= 1

    #Reset data
    data_current.pref_nan = data_whole.pref_nan.copy()

    #Blanks ids of the opposite markers
    if _marker == 'Y':
        nanCell = [np.take(data_whole.naniloc_inv[0], id_X[foldId]), np.take(data_whole.naniloc_inv[1], id_X[foldId])]
        data_current.pref_nan[nanCell] = np.nan
        data_current.updateByNan(_preDe=options.PRE_DE)

    if _marker == 'X':
        nanCell = [np.take(data_whole.naniloc_inv[0], id_Y[foldId]), np.take(data_whole.naniloc_inv[1], id_Y[foldId])]
        data_current.pref_nan[nanCell] = np.nan
        print(data_current.pref_nan)
        data_current.updateByNan(_preDe=options.PRE_DE)
    
    #Log
    markers.CURRENT_DATA = '#{}/{}, {}'.format(foldId + 1, options.K_FOLD, _marker)

    return copy.deepcopy(data_current)

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

- Data comes from 4.
------------------------------------------------------------
'''
#--Compute CF using all samples
u_dist_cf = SVD(imputation(data_whole.pref_nan, imValue=pd.read_csv(r'../data/res_demean.csv').allmean[0]))


#--Prepare pref_train, mask, and the truth
#90% X, 10% Y
#Avoid autocorrelation when using all - 1 samples predicts a target
id_X, id_Y = kFold(3, data_whole.nMN, seed=1)

#Acquire X and Y
foldId = 1
data_X = gen_preprocessing_kFold(data_whole, data_current, foldId, 'X')
data_Y = gen_preprocessing_kFold(data_whole, data_current, foldId, 'Y')
data_Y.listData()


#--Predicting Y by X with sim combinations
#Inherit
#u_dist_person  u_dist_sat  u_dist_demo  u_dist_cf  dist_triplet  dist_review  dist_genre
output = gen_learnWeight(data=data_Y, exp='1', m_dists=[np.ones((data_Y.nM, data_Y.nM)) + np.eye(data_Y.nM)], n_dists=[np.ones((data_Y.nN, data_Y.nN)) + np.eye(data_Y.nN)], _cf=False, nRef=-1, nEpoch=50, lRate=0.01, batchSize=-1, title='All')
predictions, metrics = gen_model(**output)
output = gen_learnWeight(data=data_Y, exp='1', m_dists=[u_dist_person + np.eye(data_Y.nM) * 2], n_dists=[np.ones((data_Y.nN, data_Y.nN)) + np.eye(data_Y.nN)], _cf=False, nRef=-1, nEpoch=50, lRate=0.01, batchSize=-1, title='All')
predictions, metrics = gen_model(**output)
output = gen_learnWeight(data=data_Y, exp='1', m_dists=[np.ones((data_Y.nM, data_Y.nM)) + np.eye(data_Y.nM)], n_dists=[dist_review + np.eye(data_Y.nN) * 2], _cf=False, nRef=-1, nEpoch=50, lRate=0.01, batchSize=-1, title='All')
predictions, metrics = gen_model(**output)

#Manual
options.DEBUG = False
options.DEBUG = True
predictions, metrics = gen_model(data=data_Y, exp='1', nRef=-1, m_dists=[u_dist_cf], n_dists=[np.ones((data_Y.nN, data_Y.nN))], _cf=False, m_a=[1], n_a=[1], m_b=[1], n_b=[1], c=[0], title='General model 2', target=[5, 14])
predictions, metrics = gen_model(data=data_Y, exp='1', nRef=-1, m_dists=[np.ones((data_Y.nM, data_Y.nM))], n_dists=[dist_triplet], _cf=False, m_a=[1], n_a=[1], m_b=[1], n_b=[1], c=[0], title='General model 2', target=[5, 14])




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