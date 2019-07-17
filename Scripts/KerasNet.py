import pandas as pd
import sklearn as skl
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers
from sklearn.preprocessing import OneHotEncoder as ohe, normalize
from sklearn.model_selection import train_test_split
from pathlib import Path
from math import sqrt, pow

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
    meta = normalize(meta_df.to_numpy(), axis=0, norm='max')

    data_train, data_test, meta_train, meta_test = train_test_split(data, meta, test_size=0.1)


    #get the model
    model = makeModel(data_train.shape[1])

    #train and eval
    model.fit(data_train, meta_train, epochs=500, batch_size=32, verbose = 0)
    print('eval')
    model.evaluate(data_test, meta_test, batch_size=32)

    print('predict')
    res = model.predict(data_test)
    # print('\n'.join(['{0}\t{1}'.format(x[0], x[1]) for x in zip(meta_test, res)]))
    print(rmsep(res, meta_test))

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

    
    

    




