import pandas as pd
import sklearn as skl
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers, utils, regularizers
from sklearn.preprocessing import OneHotEncoder as ohe, normalize
from sklearn.model_selection import train_test_split
from pathlib import Path
from math import sqrt, pow

import resnet

def makeModel(input_dim):
    '''
    this is the main function to make the model. start small
    '''

    model = tf.keras.Sequential()
    model.add(layers.Dense(input_dim, input_dim=input_dim, activation='relu', kernel_initializer='normal'))
    # model.add(layers.Dropout(0.4))
    # model.add(layers.Dense(128, activation='relu', kernel_initializer='normal'))
    # model.add(layers.Dropout(0.4))
    model.add(layers.Dense(64, activation='relu', kernel_initializer='normal'))
    # model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu', kernel_initializer='normal'))
    # model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, kernel_initializer='normal'))

    model.compile(optimizer=optimizers.Adam(0.005),
                loss = 'mse',
                metrics=['mae'])

    return model

def makeAutoEncoder(input_dim):
    
    # input_layer = layers.Input(shape=(input_dim,))
    # encoded = layers.Dense(32, activation='relu')(input_layer)
    # decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

    input_layer = layers.Input(shape=(input_dim,1))
    encoded = layers.Conv1D(filters=16, kernel_size=(3,), activation='relu', padding='same')(input_layer)
    encoded = layers.Conv1D(filters=16, kernel_size=(3,), activation='relu', padding='same')(encoded)
    encoded = layers.MaxPooling1D(4, padding='same')(encoded)

    decoded = layers.Conv1D(filters=16, kernel_size=(3,), activation='relu', padding='same')(encoded)
    decoded = layers.UpSampling1D(4)(decoded)
    decoded = layers.Conv1D(filters=1, kernel_size=(3,), activation='sigmoid', padding='same')(decoded)

    autoencoder = tf.keras.models.Model(input_layer, decoded)
    autoencoder.compile(optimizer=optimizers.Adagrad(lr=0.05), loss='binary_crossentropy', metrics=['accuracy'])

    autoencoder.summary()
    return autoencoder, encoded

# def small_val_reg(weight_matrix):
#     K = tf.keras.backend
#     return 0.1 / K.sum(K.abs(weight_matrix))
    



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
    encoder = ohe(sparse=False)
    data = encoder.fit_transform(df.to_numpy())
    s = data.shape
    data = data.reshape((s[0], s[1], 1))
    # data = np.pad(data, ((0,0),(0,256 - s[1])), 'constant')
    # data = data.reshape((data.shape[0], 16, 16, 1))

    # meta = normalize(meta_df.to_numpy(), axis=0, norm='max')
    meta = meta_df.to_numpy()
    meta = np.round(meta, decimals=0).astype('str')
    meta = encoder.fit_transform(meta)


    data_train, data_test, meta_train, meta_test = train_test_split(data, meta, test_size=0.1)


    # #get the model
    # input_dim = data_train.shape[1]
    # batch_size = input_dim // 10
    # autoencoder, encoder = makeAutoEncoder(input_dim)
    # model = makeModel(input_dim)
    # #train autoencoder
    # autoencoder.fit(data_train, data_train, epochs=1000, batch_size=input_dim, shuffle=True, verbose=0)
    # autoencoder.evaluate(data_train, data_train)
    # preds = autoencoder.predict(data_train)
    # print('errors: {0}'.format(np.sum(np.absolute(data_train - preds)) / data_train.shape[0] / data_train.shape[1]))


    #resnet
    model = resnet.resnet1(in_shape=(16, 16, 1), n_classes=32, opt='adam')

    # print('real\n', data_train[0].reshape((8, 29)))
    # print('pred\n', preds[0].reshape((8, 29)))

    #train and eval
    model.fit(data_train, meta_train, epochs=100, batch_size=100, shuffle=True, verbose = 1)
    print('eval')
    model.evaluate(data_test, meta_test, batch_size=100)

    # print('predict')
    # res = model.predict(data_test)
    # # print('\n'.join(['{0}\t{1}'.format(x[0], x[1]) for x in zip(meta_test, res)]))
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

    print('end!')

    
    

    




