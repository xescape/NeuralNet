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

def makeAEModel(input_dim):
    '''
    this is the main function to make the model. start small
    '''
    model = tf.keras.Sequential()

    model.add(layers.Dense(input_dim, input_dim=input_dim, activation='relu', kernel_initializer='normal'))
    # model.add(layers.Dropout(0.4))
    # model.add(layers.Dense(128, activation='relu', kernel_initializer='normal'))
    # model.add(layers.Dropout(0.4))
    model.add(layers.Dense(16, activation='relu', kernel_initializer='normal'))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(8, activation='relu', kernel_initializer='normal'))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, kernel_initializer='normal'))

    model.compile(optimizer=optimizers.Adam(0.005),
                loss = 'mse',
                metrics=['mae'])

    return model

def makeAutoEncoder(input_dim, data_train):

    def custom_cross_entropy(y_true, y_pred):
        msk = K.cast(K.equal(y_true, K.zeros_like(y_true)), 'float32')
        raw = K.binary_crossentropy(y_true, y_pred)
        base = K.mean(raw, axis=-1)
        target = 5 * K.sum(raw * msk, axis=-1) / K.sum(msk)
        final = base + target
        return final

    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(32, activation='relu')(input_layer)
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = tf.keras.models.Model(input_layer, decoded)
    autoencoder.compile(optimizer=optimizers.Nadam(lr=0.001), loss='binary_crossentropy', metrics=['binary_accuracy'])

    encoder = tf.keras.models.Model(input_layer, encoded)

    return autoencoder, encoder

def trainAEModel(data, data_train, data_test, meta_train, meta_test):
    
    input_dim = data_train.shape[1]
    autoencoder, encoder = makeAutoEncoder(input_dim, data_train)

    #train autoencoder
    autoencoder.fit(data, data, epochs=1000, batch_size=input_dim, shuffle=True, verbose=0)
    autoencoder.fit(data, data, epochs=1, batch_size=input_dim, shuffle=True, verbose=1)
    res = autoencoder.evaluate(data_test, data_test)

    base_accuracy = np.count_nonzero(data_train) / (data_train.shape[0] * data_train.shape[1])
    print('base accuracy is: {0}'.format(base_accuracy))
    real_accuracy = (res[1] - base_accuracy) / (1 - base_accuracy)
    print('real accuracy is: {0}'.format(real_accuracy))

    # #train and eval
    aemodel = makeAEModel(32)
    aemodel.fit(encoder.predict(data_train), meta_train, epochs=2000, batch_size=100, shuffle=True, verbose = 0)
    print('aemodel training loss')
    aemodel.fit(encoder.predict(data_train), meta_train, epochs=1, batch_size=100, shuffle=True, verbose = 1)
    print('aemodel eval')
    aemodel.evaluate(encoder.predict(data_test), meta_test, batch_size=data_test.shape[0])
    print('data mean = {0} and std = {1}'.format(np.mean(meta_test), np.std(meta_test)))
    return encoder, aemodel


def makeConvAutoEncoder(input_dim):
    
    # input_layer = layers.Input(shape=(input_dim,))
    # encoded = layers.Dense(32, activation='relu')(input_layer)
    # decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

    input_layer = layers.Input(shape=(input_dim,1))
    encoded = layers.Conv1D(filters=16, kernel_size=(21,), activation='relu', padding='same')(input_layer)
    # encoded = layers.Conv1D(filters=16, kernel_size=(3,), activation='relu', padding='same')(encoded)
    encoded = layers.MaxPooling1D(2, padding='same')(encoded)
    encoded = layers.Conv1D(filters=1, kernel_size=(1,), activation='relu', padding='same')(encoded)

    decoded = layers.Conv1D(filters=16, kernel_size=(1,), activation='relu', padding='same')(encoded)
    decoded = layers.UpSampling1D(2)(decoded)
    decoded = layers.Conv1D(filters=1, kernel_size=(21,), activation='sigmoid', padding='same')(decoded)

    autoencoder = tf.keras.models.Model(input_layer, decoded)
    autoencoder.compile(optimizer=optimizers.Adagrad(lr=0.02), loss='binary_crossentropy', metrics=['accuracy'])

    autoencoder.summary()
    return autoencoder, encoded


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

 
    #actual NN params
    #make the data
    raw = df.to_numpy()

    encoder = ohe(sparse=False, categories='auto')
    data = encoder.fit_transform(raw)
    s = data.shape
    print(s)
    # data = data.reshape((s[0], s[1], 1)) #this is for the conv autoencoder
    
    # data = np.pad(data, ((0,0),(0,256 - s[1])), 'constant')
    # data = data.reshape((data.shape[0], 16, 16, 1))

    # meta = normalize(meta_df.to_numpy(), axis=0, norm='max')
    meta = meta_df.to_numpy()
    # meta = np.round(meta, decimals=0).astype('str')
    # meta = encoder.fit_transform(meta)

    # #base model
    # data_train, data_test, meta_train, meta_test = train_test_split(data, meta, test_size=0.25)
    # trainModel(data_train, data_test, meta_train, meta_test)

    #aemodel
    # train_aemodel(data, data_train, data_test, meta_train, meta_test)
    # data_train, data_test, meta_train, meta_test = train_test_split(data, meta, test_size=0.25)

    
    #prefilter
    n_f = 256
    filtered_data, data_idx = prefiltering(data, meta, n_f)
    
    #for if you want to do the candidate genes again
    # reverse_encoding = revOneHot(raw, data)
    # original_idx = [reverse_encoding[x] for x in data_idx]

    # print('good indices')
    # print(original_idx)

    # print('second lasso ranking')
    # second_idx = trainFakeLasso(filtered_data, meta)
    # try:
    #     second_idx_fixed = [original_idx[x] for x in second_idx]
    # except:
    #     print(len(second_idx), second_idx)
    #     print(len(original_idx), second_idx)
    n_folds = 5
    kf = KFold(n_splits=n_folds)

    # f_data_train, f_data_test, f_meta_train, f_meta_test = train_test_split(filtered_data, meta, test_size=0.25)
    # data_train, data_test, meta_train, meta_test = train_test_split(data, meta, test_size=0.25)

    # np.savez(data_out_path, idx = original_idx, data = filtered_data)
    for train_index, test_index in kf.split(data):

########
        f_data_train, f_data_test = filtered_data[train_index], filtered_data[test_index]
        data_train, data_test = data[train_index], data[test_index]
        meta_train, meta_test = meta[train_index], meta[test_index]
###########
        outpath = in_path / 'results'
        lasso_filtered_results, sorted_idx = trainLasso(f_data_train, f_data_test, meta_train, meta_test)
        write_result(meta_test, lasso_filtered_results, outpath / 'lasso_filtered.tsv')

        lasso_results, sorted_idx = trainLasso(data_train, data_test, meta_train, meta_test)
        write_result(meta_test, lasso_results, outpath / 'lasso_nofilter.tsv')
    ###########
        
        print('dense filtered')
        prefilter_results = trainPrefilterModel(f_data_train, f_data_test, meta_train, meta_test, log_path, model_path)
        write_result(meta_test, prefilter_results, outpath / 'dense_filtered.tsv')

        # print('dense nofilter')
        # nofilter_results = trainPrefilterModel(data_train, data_test, meta_train, meta_test, log_path, model_path)
        # write_result(meta_test, nofilter_results, outpath / 'dense_nofilter.tsv')

    #############
    #AEMODEL
        # print('ae filter')
        # model = trainAEModel(data, f_data_train, f_data_test, meta_train, meta_test)
        # ae_results = model.predict(f_data_test)
        # write_result(meta_test, ae_results, in_path / 'ae_filtered_result.tsv')

        # print('ae nofilter')
        # encoder, model = trainAEModel(data, data_train, data_test, meta_train, meta_test)
        # ae_results = model.predict(encoder.predict(data_test))
        # write_result(meta_test, ae_results, outpath / 'ae_nofilter.tsv')

    ###########
    #resnet
        # sf_train = f_data_train.shape
        # sf_test = f_data_test.shape

        s_train = f_data_train.shape
        s_test = f_data_test.shape

        # f_data2d_train = f_data_train.reshape((sf_train[0], sf_train[1], 1))
        # f_data2d_test = f_data_test.reshape((sf_test[0], sf_test[1], 1))

        # print(f_data2d_train.shape)
        # f_data2d_train = np.pad(f_data2d_train, ((0,0),(0,256 - sf_train[1])), 'constant')
        # f_data2d_train = f_data2d_train.reshape((-1, 16, 16, 1))

        # f_data2d_test = np.pad(f_data2d_test, ((0,0),(0,256 - sf_test[1])), 'constant')
        # f_data2d_test = f_data2d_test.reshape((-1, 16, 16, 1))

        # model = resnet.resnet1(in_shape=(64, 64, 1), n_classes=1, opt='Nadam')
        # print('resnet filtered')
        # model.fit(f_data2d_train, meta_train, epochs=2000, batch_size=20, shuffle=True, validation_split=0.25, verbose=0)
        # res_res = model.predict(f_data2d_test)
        # write_result(meta_test, res_res, in_path / 'res_filtered_result.tsv')

    ###
        #Resnet filtered
        # f_data2d_train = f_data_train.reshape((s_train[0], s_train[1], 1))
        # f_data2d_test = f_data_test.reshape((s_test[0], s_test[1], 1))

        
        # f_data2d_train = np.pad(f_data2d_train, ((0,0),(0,256 - s_train[1]), (0,0)), 'constant')
        # f_data2d_train = f_data2d_train.reshape((-1, 16, 16, 1))

        # f_data2d_test = np.pad(f_data2d_test, ((0,0),(0,256 - s_test[1]), (0,0)), 'constant')
        # f_data2d_test = f_data2d_test.reshape((-1, 16, 16, 1))

        # model = resnet.resnet1(in_shape=(16, 16, 1), n_classes=1, opt='Nadam')
        # print('resnet filter')
        # model.fit(f_data2d_train, meta_train, epochs=1000, batch_size=20, shuffle=True, validation_split=0.25, verbose=0)
        # res_res = model.predict(f_data2d_test)
        # write_result(meta_test, res_res, outpath / 'res_filtered.tsv')


def rmsep(x, y):
    return np.mean(np.sqrt(np.power(x - y, 2))) / np.mean(y)


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


def revOneHot(input, encoded):

    n_vecs = input.max(axis=0) + 1
    cumulative = np.cumsum(n_vecs) - 1

    res = []
    for x in range(encoded.shape[1]):
        if x == 0:
            res.append(0)
            continue
        
        proxy = 1 / (x - cumulative)
        msk = proxy < 0
        proxy = proxy * msk
        res.append(proxy.argmin())
    
    return res

if __name__ == "__main__":

    #paths
    # in_path = Path('/d/data/plasmo/newsim')

    # in_path = Path('D:\\Documents\\data\\plasmo\\newsim')

    # #toy data
    # in_path = Path('D:\\Documents\\data\\plasmo\\newsim')
    # paintings_path = in_path / 'painting.tsv'
    # meta_path = in_path / 'sim_meta.tsv'

    # for realsies
    #for the 150 sample dataset
    # in_path = Path('D:\\Documents\\data\\plasmo\\training_data')
    # paintings_path = in_path / 'plasmo5k_nn.tsv'
    # meta_path = in_path / 'meta_v2.txt'

    #for nat_com data
    #defaults to running on scinet beluge in case I forget
    in_path = Path('/home/xescape/scratch/nn')
    try:
        if sys.argv[1] == 'local':
            in_path = Path('D:\\Documents\\data\\plasmo\\training_data')
    except:
        continue

    paintings_path = in_path / 'plasmo2.tsv'
    meta_path = in_path / 'meta_v2.txt'

    #make the log folder if it's not there
    if not is_dir(in_path / 'nn_logs'):
        os.mkdir(in_path / 'nn_logs')

    log_path = in_path / 'nn_logs' / 'prefilter2k'
    model_path = in_path / 'nn_logs' / 'curr_model5.h5'
    data_out_path = in_path / 'nn_logs' / 'curr_data5.npz'


    df, meta_df = importData(paintings_path, meta_path)
    run(df, meta_df, in_path, log_path, model_path, data_out_path)

    print('end!')

    
    

    




