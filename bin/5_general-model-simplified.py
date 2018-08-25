'''
------------------------------------------------------------
Component functions

- Replace the original functions in 4.
------------------------------------------------------------
'''
def gen_preprocessing_kFold(foldId, _marker):
    assert _marker in ['X', 'Y'], 'Wrong marker.'

    #Fold id 1 -> 0. Fold ID starts from 1.
    foldId -= 1

    #Manage global directly
    global pref_nan, prefs, nM, nN, nMN, isnan_inv, gameRatedByRater

    #Reset data
    pref_nan, prefs, nM, nN, nMN, isnan_inv, gameRatedByRater = preprocessing(description=False, _preDe=True)
    naniloc_inv = np.where(isnan_inv)

    #Blanks ids of the opposite markers
    if _marker == 'Y':
        nanCell = [np.take(naniloc_inv[0], id_X[foldId]), np.take(naniloc_inv[1], id_X[foldId])]
        pref_nan[nanCell] = np.nan

    if _marker == 'X':
        nanCell = [np.take(naniloc_inv[0], id_Y[foldId]), np.take(naniloc_inv[1], id_Y[foldId])]
        pref_nan[nanCell] = np.nan

    #Update vars
    return (pref_nan, *preprocessing_core(pref_nan, _preDe=True))


def gen_pref8mask(m, n, _colMask):

    #--Pref
    #Use Y
    pref_nan, prefs, nM, nN, nMN, isnan_inv, gameRatedByRater = data_Y

    #Get the truth
    truth = pref_nan[m, n]
    

    #--Mask
    #Use X
    pref_nan, prefs, nM, nN, nMN, isnan_inv, gameRatedByRater = data_X

    #Mask the entire column (simulate a new product which has no rating)
    if _colMask: pref_nan[:, n] = np.nan
    
    #Remove the entire column from the matrix
    if _colMask: isnan_inv[:, n] = False


    return pref_nan, isnan_inv, truth
    



'''
------------------------------------------------------------
Model
------------------------------------------------------------
'''
#--Pre-removing col and row effects
#Read from file, processing with all samples
pref_nan, prefs, nM, nN, nMN, isnan_inv, gameRatedByRater = preprocessing(description=False, _preDe=True)


#--Compute CF using all samples
u_dist_cf = SVD(imputation(pref_nan, imValue=pd.read_csv(r'../data/res_demean.csv').allmean[0]))


#--Prepare pref_train, mask, and the truth
#90% X, 10% Y
#Avoid autocorrelation when using all - 1 samples predicts a target
id_X, id_Y = kFold(10, nMN, seed=1)

#Acquire X and Y
foldId = 1
data_X = gen_preprocessing_kFold(foldId, 'X')
data_Y = gen_preprocessing_kFold(foldId, 'Y')


#--Predicting Y by X with sim combinations
#Set global to Y
pref_nan, prefs, nM, nN, nMN, isnan_inv, gameRatedByRater = data_Y

#Inherit
#u_dist_person  u_dist_sat  u_dist_demo  u_dist_cf  dist_triplet  dist_review  dist_genre
output = gen_learnWeight(exp='1', m_dists=[u_dist_cf], n_dists=[np.ones((nN, nN))], _cf=False, _colMask=False, nRef=-1, nEpoch=50, lRate=0.1, batchSize=-1, title='All')
predictions, metrics = gen_model(**output)

#Manual
predictions, metrics = gen_model(exp='1', nRef=-1, m_dists=[], n_dists=[dist_review], _cf=False, _colMask=True, m_a=[1], n_a=[0.7599], m_b=[1], n_b=[1], c=[4.104656], title='General model 2')




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