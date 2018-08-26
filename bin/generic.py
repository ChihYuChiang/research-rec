#--Universal container
class UniversalContainer():

    def listData(self):
        data = [(item, getattr(self, item)) for item in dir(self) if not callable(getattr(self, item)) and not item.startswith("__")]
        for item in data: print(item)

    def listMethod(self):
        print([item for item in dir(self) if callable(getattr(self, item)) and not item.startswith("__")])


#--Setting container
class SettingContainer(UniversalContainer):

    def update(self, **kwarg):
        for key, value in kwarg.items():
            setattr(self, key, value)

    __init__ = update


#--Flatten list (recursive)
#Parameter: l, a list
#Return: a flattened list as a generator
def flattenList(l):
    import collections

    for el in l:
        if isinstance(el, collections.Sequence) and not isinstance(el, (str, bytes)):
            yield from flattenList(el)
        else:
            yield el


#--Element-wise list operation
#Return: operated list
def listEWiseOp(op, l):
    return list(map(op, l))


#--Graphing 2-D scatter plot
#With distribution and linear fitting line
def scatter(vectors, names):

    sns.set(color_codes=True)

    df = pd.DataFrame({
        names[0]: vectors[0],
        names[1]: vectors[1]
    })

    g = sns.jointplot(x=names[0], y=names[1], data=df, color="m", kind="reg", scatter_kws={"s": 10})
    plt.show()


#--Remove row and column effect
#Acquire row and column effect
def getMean(matrix):

    #Compute row and column effects
    mMean = np.nanmean(matrix, axis=1) - np.mean(np.nanmean(matrix, axis=1))
    nMean = np.nanmean(matrix, axis=0) - np.mean(np.nanmean(matrix, axis=0))

    #Deal with empty row/column (all nan)
    mMean[np.where(np.isnan(mMean))] = np.nanmean(matrix)
    nMean[np.where(np.isnan(nMean))] = np.nanmean(matrix)

    return mMean, nMean


#--Evaluate model with mse, cor, and graphing
def evalModel(predictions, truth, nMN, title, graph, logger=None):

    #Description
    mse = np.sum(np.square(predictions - truth)) / nMN
    cor = np.corrcoef(predictions, truth)[0, 1]
    rho, _ = sp.stats.spearmanr(predictions, truth)

    output = logger if logger else print
    output('-' * 60)
    output(title)
    output('MSE = {}'.format(mse))
    output('Correlation = {}'.format(cor))
    output('RankCorrelation = {}'.format(rho))

    #Graphing
    if graph: scatter([truth, predictions], ['truth', 'predictions'])

    return mse, cor, rho


#--Acquire ids of a k-fold training testing set
def kFold(k, nMN, seed=1):

    #Reset the seed
    np.random.seed(seed=seed)

    #The indice to be selected
    rMN = np.arange(nMN)
    np.random.shuffle(rMN)

    #Indicator
    #To make sure the distribution is as evenly as possible
    ind = abs(nMN - (nMN // k + 1) * (k - 1) - (nMN // k + 1)) < abs(nMN - (nMN // k) * (k - 1) - (nMN // k))

    #Series id based on k
    anchor = np.arange(k) * (nMN // k + ind)
    
    #Acquire the training and testing set ids
    id_test = [rMN[anchor[i]:(anchor[i + 1] if i + 1 != len(anchor) else None)] for i in range(len(anchor))]
    id_train = [np.setdiff1d(rMN, id_test[i]) for i in range(len(id_test))]

    #Deal with 1 fold (using entire dataset to train and test)
    if k == 1: id_train = id_test

    return id_train, id_test
    

#--Logger
def iniLogger(loggerName, fileName, _console):

    #Use the default logger
    logger = logging.getLogger(loggerName)
    logger.setLevel(logging.DEBUG)

    #Create formatter
    formatter = logging.Formatter('%(message)s')

    #Create file handler and add to logger
    fh = logging.FileHandler('../log/{}'.format(fileName), mode='w+')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    #Create console handler and add to logger
    if _console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


#--nan imputation by total mean and adjust by column and row effects
#Return imputed matrix
def imputation(matrix, imValue=None):
    
    #Find nan iloc
    naniloc = np.where(np.isnan(matrix))

    #Insert appropriate value into the matrix where is nan
    #Impute a predefined value
    if imValue: matrix[naniloc] = imValue
        
    #Impute overall mean
    else:
        #Compute column and row effect
        mMean, nMean = getMean(matrix)

        #np.take is faster than fancy indexing i.e. nMean[[1, 3, 5]]
        matrix[naniloc] = np.nanmean(matrix) + np.take(nMean, naniloc[1]) + np.take(mMean, naniloc[0])

        #Substract mean, col and row effects from pref
        matrix = deMean(matrix, mMean, nMean)

    return matrix