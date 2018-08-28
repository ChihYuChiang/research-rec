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
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    sns.set(color_codes=True)

    df = pd.DataFrame({
        names[0]: vectors[0],
        names[1]: vectors[1]
    })

    g = sns.jointplot(x=names[0], y=names[1], data=df, color="m", kind="reg", scatter_kws={"s": 10})
    plt.show()


#--Evaluate model with mse, cor, and graphing
def evalModel(predictions, truth, nMN, title, graph, logger=None):
    import numpy as np
    import scipy as sp

    #Description
    mse = np.sum(np.square(predictions - truth)) / nMN
    cor = np.corrcoef(predictions, truth)[0, 1]
    rho, _ = sp.stats.spearmanr(a=predictions, b=truth)

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
    import numpy as np

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
    import logging

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