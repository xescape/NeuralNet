import sys
import pandas as pd
import sklearn as skl
import tensorflow as tf
import numpy as np
import datetime
import logging
from tensorflow.keras import layers, optimizers, utils, regularizers, backend as K
from sklearn.preprocessing import OneHotEncoder as ohe, normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error as mse
from pathlib import Path
from math import sqrt, pow
from sklearn.model_selection import KFold
import random
import autokeras as ak


def testAutoKEras():
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    input_layer = ak.ImageInput(shape=(28,28,1))
    regression = ak.ClassificationHead(metrics=['mae'])

    automodel = ak.AutoModel(inputs=[input_layer], outputs=[regression])
    automodel.fit(x_train, y_train, epochs=10000, callbacks=[tf.keras.callbacks.EarlyStopping()], validation_data=(x_test, y_test))

    # x_image = x_train.reshape(x_train.shape + (1,))
    # x_test = x_test.reshape(x_test.shape + (1,))

    # x_structured = np.random.rand(x_train.shape[0], 100)
    # y_regression = np.random.rand(x_train.shape[0], 1)

    # # Build model and train.
    # automodel = ak.AutoModel(
    # inputs=[ak.ImageInput()],
    # outputs=[ak.RegressionHead(metrics=['mae'])])
    # automodel.fit([x_image],
    #             [y_regression],
    #             validation_split=0.2)

def makeAutoKeras(x_train, y_train, x_test, y_test, n, directory, logger):
    #l is the length of each vector

    k = x_train.shape[0]
    l = x_train.shape[1]
    m = x_test.shape[0]

    x_train = x_train.reshape((k, l, 1, 1))
    x_test = x_test.reshape((m, l, 1, 1))

    input_layer = ak.ImageInput(shape=(l,1,1))
    regression = ak.RegressionHead(metrics=['mae'])

    automodel = ak.AutoModel(inputs=[input_layer], outputs=[regression])
    automodel.fit(x_train, y_train, epochs=10000, callbacks=[], validation_data=(x_test, y_test))

    error = mse(y_test, automodel.predict(x_test))
    logger.info('Model {0} got MSE of {1}'.format(n, error))

def prefiltering(data, meta, n):
    #make a linear model and get rid of the bottom stuff
    model = LassoCV(cv = 3)
    model.fit(data, meta.reshape((meta.shape[0])))
    coefs = np.abs(model.coef_)
    # coefs = coefs.reshape((coefs.shape[1],))
    cutoff = max(sorted(coefs)[n], 0.000000000001) #minimum coefficient
    good_idx = np.where((coefs >= cutoff))[0]
    res = data[:,good_idx]
    print('{0} features selected'.format(len(good_idx)))
    return res, good_idx

def importData(paintings_path, meta_path):
    '''
    imports the chromsome painting and meta data
    '''
    #for the paintings
    # if alt:
    #     df = read
    df = pd.read_csv(paintings_path, sep='\t', header=None, index_col=0)
    
    #for the meta
    meta_df = pd.read_csv(meta_path, sep='\t', header=0, index_col=0)
    
    df, meta_df = df.align(meta_df, axis=0, join='inner')
    
    return df, meta_df


def run(df, meta_df, in_path, log_path, model_path):

    logger = configLogger(log_path)


    #actual NN params
    #make the data
    raw = df.to_numpy()
    encoder = ohe(sparse=False, categories='auto')
    data = encoder.fit_transform(raw)
    meta = meta_df.to_numpy()

    n_f = 512
    filtered_data, data_idx = prefiltering(data, meta, n_f)

    n_folds = 5
    n = 1
    kf = KFold(n_splits=n_folds)
    for train_index, test_index in kf.split(data):

        data_train, data_test = filtered_data[train_index], filtered_data[test_index]
        meta_train, meta_test = meta[train_index], meta[test_index]

        makeAutoKeras(data_train, meta_train, data_test, meta_test, n, model_path, logger)

        n += 1 #put the models in different folders

def write_result(meta_test, result, out_path):

    if out_path.is_file():
        header = False
    else:
        header = True

    if not out_path.parent.is_dir():
        out_path.parent.mkdir(parents=True)

    with open(out_path, 'a') as output:
        if header:
            output.write('real\tpredict\n')
        else:
            output.write('\n')

        main_text = ['{0}\t{1}'.format(pair[0], pair[1]) for pair in zip(list(meta_test.reshape((-1,))), list(result.reshape((-1,))))]
        output.write('\n'.join(main_text))

def configLogger(_path):
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s:\n%(message)s\n')
    
    fh = logging.FileHandler(_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

if __name__ == "__main__":

    #for nat_com data
    #defaults to running on scinet beluge in case I forget
    in_path = Path('/home/xescape/scratch/nn')
    try:
        if sys.argv[1] == 'local':
            in_path = Path('D:\\Documents\\data\\plasmo\\training_data')
    except:
        pass
#nat only
    # paintings_path = in_path / 'nat_nn.tsv'
    # meta_path = in_path / 'meta.tsv'

    paintings_path = in_path / 'plasmo5k_nn.tsv'
    meta_path = in_path / 'meta_v2.txt'

    #make the log folder if it's not there
    log_path = in_path / 'nn_logs'
    # if not log_path.is_dir():
    #     log_path.mkdir()
    
    model_path = in_path / 'models'
    if not model_path.is_dir():
        model_path.mkdir()

    df, meta_df = importData(paintings_path, meta_path)
    run(df, meta_df, in_path, log_path, model_path)

    print('end!')

    
    

    




