import sys
import pandas as pd
import sklearn as skl
import tensorflow as tf
import numpy as np
import datetime
from tensorflow.keras import layers, optimizers, utils, regularizers, backend as K
from sklearn.preprocessing import OneHotEncoder as ohe, normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from pathlib import Path
from math import sqrt, pow
from sklearn.model_selection import KFold
import resnet
import random

def trainLasso(data_train, data_test, meta_train, meta_test):
    model = LassoCV(cv = 5)
    model.fit(data_train, meta_train.reshape((meta_train.shape[0])))
    print('lasso model r2 = {0}'.format(model.score(data_test, meta_test.reshape((meta_test.shape[0])))))

    coefs = np.abs(model.coef_)
    sorted_idx = sorted(range(coefs.shape[0]), key = lambda x: coefs[x], reverse = True)

    return model.predict(data_test), sorted_idx

def trainFakeLasso(data, meta):
    model = LassoCV(cv = 5)
    model.fit(data, meta.reshape((meta.shape[0])))
    print('lasso model r2 = {0}'.format(model.score(data, meta.reshape((meta.shape[0])))))
    coefs = np.abs(model.coef_)
    sorted_idx = sorted(range(coefs.shape[0]), key = lambda x: coefs[x], reverse = True)

    return sorted_idx

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


def makeModel(input_dim):
    '''
    this is the main function to make the model. start small
    '''
    model = tf.keras.Sequential()
    model.add(layers.Dense(input_dim, input_dim=input_dim, activation='relu', kernel_initializer='normal'))
    model.add(layers.Dense(64, activation='relu', kernel_initializer='normal'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='relu', kernel_initializer='normal'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation='relu', kernel_initializer='normal'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, kernel_initializer='normal'))

    model.compile(optimizer=optimizers.Nadam(),
                loss = 'mse',
                metrics=['mae'])

    return model

def trainModel(data_train, data_test, meta_train, meta_test):
    input_dim = data_train.shape[1]
    model = makeModel(input_dim)
    model.fit(data_train, meta_train, epochs=3000, batch_size=5, shuffle=True, verbose=0)
    print('base model training loss')
    model.fit(data_train, meta_train, epochs=1, batch_size=5, shuffle=True, verbose=1)
    print('base model eval')
    model.evaluate(data_test, meta_test, batch_size=data_test.shape[1])
    print('data mean = {0} and std = {1}'.format(np.mean(meta_test), np.std(meta_test)))

def makePrefilterModel(input_dim):
    '''
    this is the main function to make the model. start small
    '''
    model = tf.keras.Sequential()

    model.add(layers.Dense(input_dim, input_dim=input_dim, activation='relu', kernel_initializer='normal'))
    model.add(layers.Dense(128, activation='relu', kernel_initializer='normal'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu', kernel_initializer='normal'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='relu', kernel_initializer='normal'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation='relu', kernel_initializer='normal'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, kernel_initializer='normal'))

    model.compile(optimizer=optimizers.Nadam(),
                loss = 'mse',
                metrics=['mae'])

    return model

def makeSimplePrefilterModel(input_dim):
    '''
    this is the main function to make the model. start small
    '''
    model = tf.keras.Sequential()

    model.add(layers.Dense(input_dim, input_dim=input_dim, activation='relu', kernel_initializer='normal'))
    # model.add(layers.Dense(128, activation='relu', kernel_initializer='normal'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(64, activation='relu', kernel_initializer='normal'))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='relu', kernel_initializer='normal'))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation='relu', kernel_initializer='normal'))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, kernel_initializer='normal'))

    model.compile(optimizer=optimizers.Nadam(),
                loss = 'mse',
                metrics=['mae'])

    return model

def trainPrefilterModel(data_train, data_test, meta_train, meta_test, log_path, model_path):
    input_dim = data_train.shape[1]
    model = makePrefilterModel(input_dim)
    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir=str(log_path))
    model.fit(data_train, meta_train, epochs=2000, batch_size=input_dim, shuffle=True, validation_split=0.25, verbose=0) #no tensorboard
    print('prefilter model training loss')
    model.fit(data_train, meta_train, epochs=1, batch_size=20, shuffle=True, verbose=1)
    print('prefilter model eval')
    model.evaluate(data_test, meta_test, batch_size=data_test.shape[1])
    print('data mean = {0} and std = {1}'.format(np.mean(meta_test), np.std(meta_test)))
    # model.save(model_path)

    return model.predict(data_test)

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


def run(df, meta_df, in_path, log_path, model_path, data_out_path):

    print('Input shape is {0}'.format(str(df.shape)))

    data = df.to_numpy()
    meta = meta_df.to_numpy()
        
    #prefilter
    #try to reuse prefiltering results, but will redo if it didn't work
    prefilter_path = in_path / 'prefilter.txt'
    try:
        with open(prefilter_path, 'r') as input:
            data = input.read()
        
        lines = [(e[0], e[1]) for e in [f.split('\t') for f in data.strip('\n').split('\n')]]
        filtered_data, data_idx = zip(*lines)
    except:
        n_f = 256
        filtered_data, data_idx = prefiltering(data, meta, n_f)
        with open(prefilter_path, 'w') as output:
            lines = ['\t'.join(f, d) for f, d in zip(filtered_data, data_idx)]
            output.write('\n'.join(lines))

    n_folds = 5
    kf = KFold(n_splits=n_folds)
    out_path = in_path / 'results'
    if not out_path.is_dir():
        out_path.mkdir()

    for train_index, test_index in kf.split(data):

        f_data_train, f_data_test = filtered_data[train_index], filtered_data[test_index]
        data_train, data_test = data[train_index], data[test_index]
        meta_train, meta_test = meta[train_index], meta[test_index]

        print('lasso filtered')
        lasso_filtered_results, sorted_idx = trainLasso(f_data_train, f_data_test, meta_train, meta_test)
        write_result(meta_test, lasso_filtered_results, out_path / 'lasso_filtered.tsv')
        
        print('lasso not filtered')
        lasso_results, sorted_idx = trainLasso(data_train, data_test, meta_train, meta_test)
        write_result(meta_test, lasso_results, out_path / 'lasso_nofilter.tsv')
        
        print('dense filtered')
        prefilter_results = trainPrefilterModel(f_data_train, f_data_test, meta_train, meta_test, log_path, model_path)
        write_result(meta_test, prefilter_results, out_path / 'dense_filtered.tsv')

        print('dense not filtered')
        nofilter_results = trainPrefilterModel(data_train, data_test, meta_train, meta_test, log_path, model_path)
        write_result(meta_test, nofilter_results, out_path / 'dense_nofilter.tsv')

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

if __name__ == "__main__":

    #for the new transformed data
    #defaults to running on scinet beluge in case I forget
    in_path = Path('/home/xescape/scratch/nn_scalar')
    try:
        if sys.argv[1] == 'local':
            in_path = Path('D:\\Documents\\data\\plasmo\\nn_scalar')
    except:
        continue

    data_path = in_path / 'combined_scores.tsv'
    meta_path = in_path / 'meta.tsv'
    log_dir = in_path / 'nn_logs'

    #make the log folder if it's not there
    if not log_dir.is_dir():
        log_dir.mkdir()

    log_path = in_path / 'nn_logs' / 'logs'
    model_path = in_path / 'nn_logs' / 'curr_model5.h5'
    data_out_path = in_path / 'nn_logs' / 'curr_data5.npz'


    df, meta_df = importData(data_path, meta_path)
    run(df, meta_df, in_path, log_path, model_path, data_out_path)

    print('end!')

    
    

    




