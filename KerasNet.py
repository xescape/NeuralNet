import sys
import pandas as pd
import sklearn as skl
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers, utils, regularizers, backend as K
from sklearn.preprocessing import OneHotEncoder as ohe, normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from pathlib import Path
from math import sqrt, pow

import resnet

def prefiltering(data, meta, n):
    #make a linear model and get rid of the bottom stuff

    model = LassoCV()
    model.fit(data, meta.reshape((meta.shape[0])))
    coefs = np.abs(model.coef_)
    # coefs = coefs.reshape((coefs.shape[1],))
    cutoff = max(sorted(coefs)[n], 0.000001) #minimum coefficient

    good_idx = np.where((coefs >= cutoff))[0]
    res = data[:,good_idx]
    return res


def makeModel(input_dim):
    '''
    this is the main function to make the model. start small
    '''
    model = tf.keras.Sequential()
    model.add(layers.Dense(input_dim, input_dim=input_dim, activation='relu', kernel_initializer='normal'))
    model.add(layers.Dense(32, activation='relu', kernel_initializer='normal'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(16, activation='relu', kernel_initializer='normal'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, kernel_initializer='normal'))

    model.compile(optimizer=optimizers.Adam(0.005),
                loss = 'mse',
                metrics=['mae'])

    return model

def trainModel(data_train, data_test, meta_train, meta_test):
    input_dim = data_train.shape[1]
    model = makeModel(input_dim)
    model.fit(data_train, meta_train, epochs=10000, batch_size=20, shuffle=True, verbose=0)
    print('base model training loss')
    model.fit(data_train, meta_train, epochs=1, batch_size=input_dim, shuffle=True, verbose=1)
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
    model.add(layers.Dense(12, activation='relu', kernel_initializer='normal'))
    model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(12, activation='relu', kernel_initializer='normal'))
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
    autoencoder.fit(data, data, epochs=3000, batch_size=input_dim, shuffle=True, verbose=0)
    autoencoder.fit(data, data, epochs=1, batch_size=input_dim, shuffle=True, verbose=1)
    res = autoencoder.evaluate(data_test, data_test)

    base_accuracy = np.count_nonzero(data_train) / (data_train.shape[0] * data_train.shape[1])
    print('base accuracy is: {0}'.format(base_accuracy))
    real_accuracy = (res[1] - base_accuracy) / (1 - base_accuracy)
    print('real accuracy is: {0}'.format(real_accuracy))

    # #train and eval
    aemodel = makeAEModel(32)
    aemodel.fit(encoder.predict(data_train), meta_train, epochs=3000, batch_size=100, shuffle=True, verbose = 0)
    print('aemodel training loss')
    aemodel.fit(encoder.predict(data_train), meta_train, epochs=1, batch_size=100, shuffle=True, verbose = 1)
    print('aemodel eval')
    aemodel.evaluate(encoder.predict(data_test), meta_test, batch_size=data_test.size[0])
    print('data mean = {0} and std = {1}'.format(np.mean(meta_test), np.std(meta_test)))


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
    model.add(layers.Dense(24, activation='relu', kernel_initializer='normal'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation='relu', kernel_initializer='normal'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(8, activation='relu', kernel_initializer='normal'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, kernel_initializer='normal'))

    model.compile(optimizer=optimizers.Nadam(),
                loss = 'mse',
                metrics=['mae'])

    return model

def trainPrefilterModel(data_train, data_test, meta_train, meta_test):
    input_dim = data_train.shape[1]
    model = makeModel(input_dim)
    model.fit(data_train, meta_train, epochs=10000, batch_size=20, shuffle=True, verbose=0)
    print('prefilter model training loss')
    model.fit(data_train, meta_train, epochs=1, batch_size=20, shuffle=True, verbose=1)
    print('prefilter model eval')
    model.evaluate(data_test, meta_test, batch_size=data_test.shape[1])
    print('data mean = {0} and std = {1}'.format(np.mean(meta_test), np.std(meta_test)))

def importData(paintings_path, meta_path):
    '''
    imports the chromsome painting and meta data
    '''
    #for the paintings
    df = pd.read_csv(paintings_path, sep='\t', header=None, index_col=0)
    
    #for the meta
    meta_df = pd.read_csv(meta_path, sep='\t', header=0, index_col=0)
    
    df, meta_df = df.align(meta_df, axis=0)
    
    return df, meta_df

def run(df, meta_df):

 
    #actual NN params

    #make the data
    encoder = ohe(sparse=False, categories='auto')
    data = encoder.fit_transform(df.to_numpy())
    s = data.shape
    # data = data.reshape((s[0], s[1], 1)) #this is for the conv autoencoder
    
    # data = np.pad(data, ((0,0),(0,256 - s[1])), 'constant')
    # data = data.reshape((data.shape[0], 16, 16, 1))

    # meta = normalize(meta_df.to_numpy(), axis=0, norm='max')
    meta = meta_df.to_numpy()
    # meta = np.round(meta, decimals=0).astype('str')
    # meta = encoder.fit_transform(meta)

    # #base model
    data_train, data_test, meta_train, meta_test = train_test_split(data, meta, test_size=0.25)
    trainModel(data_train, data_test, meta_train, meta_test)

    #aemodel
    # train_aemodel(data, data_train, data_test, meta_train, meta_test)
    
    #prefilter
    n_f = 32
    filtered_data = prefiltering(data, meta, n_f)
    data_train, data_test, meta_train, meta_test = train_test_split(filtered_data, meta, test_size=0.25)
    trainPrefilterModel(data_train, data_test, meta_train, meta_test)

    # #resnet
    # model = resnet.resnet1(in_shape=(16, 16, 1), n_classes=32, opt='adam')

    # print('real\n', data_train[0].reshape((8, 29)))
    # print('pred\n', preds[0].reshape((8, 29)))



    # print('predict')
    # res = model.predict(data_test)
    # print('\n'.join(['{0}\t{1}'.format(x[0], x[1]) for x in zip(meta_test, res)]))
    # print(rmsep(res, meta_test))

    return

def rmsep(x, y):
    return np.mean(np.sqrt(np.power(x - y, 2))) / np.mean(y)

if __name__ == "__main__":

    #paths
    in_path = Path('/d/data/plasmo/newsim')
    paintings_path = in_path / 'painting.tsv'
    meta_path = in_path / 'sim_meta.tsv'

    df, meta_df = importData(paintings_path, meta_path)
    run(df, meta_df)

    # prefilter_test()

    print('end!')

    
    

    




