'''
Created on Nov 16, 2018

@author: Javi
'''

import tensorflow as tf
# import keras
# import keras.layers as layers
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import ImportData as id
import numpy as np
import pandas
import sys
import os
import logging
import pathlib
from datetime import datetime
import time
import re
# def op_model(features, labels, mode, params):
#     
#     model = keras.Sequential()
#     
#     
#     #input???
#     
#     model.add(layers.Dense(32, input_dim = input_dim))
#     model.add(layers.Activation('relu'))
#     
#     model.add(layers.Dense(10))
#     model.add
#     
#     
#     return

def loadConfig(config_path):
    '''simple load config. only get lines that are like ^X=Y\n'''
    
    line_pattern = '^[^#]+?[=].+?'
    
    res = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if re.match(line_pattern, line):
                split = line.split('=')
                try:
                    res[split[0].strip()] = float(split[1].strip())
                except:
                    res[split[0].strip()] = split[1].strip()

    return res
    
    
def getLogger(name, path):
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: \n %(message)s \n')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    
    return logger
    
def newData(popnet_dir, meta_path, key_path, data_path, label_path):
    '''
    so we plan on only doing data manipulation when we're changing the data
    and as such we're gonna pickle the dataframes, and recover them to test new
    models.
    to refresh the data, delete the folder containing the pickled files.
    '''
   
    data, labels, max = id.importData(popnet_dir, meta_path, key_path)
    target = 'clearance half-life'
    labels = labels.filter(items=['Sample_Name', target])
    labels = labels.apply(pandas.to_numeric, errors='coerce')
    
       
    #specific data processing
    #we're keeping each row to be a feature, and column to be a sample.
    col_labels = data.apply(lambda row: concat_chr_pos(row['Chromosome'], row['Position']), axis=1)
    data.index = col_labels
    data = data[data.columns[3:]]
    data = data.T
    
    data = data.join(labels)
    
    data = data[data[target].notnull()]
    data = data.sample(frac = 1) #shuffles the rows.

    labels = data.filter(items=[target])
    data = data.filter(items=data.columns.difference([target]))
    
    data.to_pickle(data_path)
    labels.to_pickle(label_path)
    
    return

def getData(data_path, label_path):
    '''
    recovers the pickled dataframes and recomputs the highest number
    '''
    data = pandas.read_pickle(data_path)
    labels = pandas.read_pickle(label_path)
        
    max = int(data.max().max() + 2) #it's hard to account for grey and black so I'm just going to manually add 2. shouldn't be a big deal anyways.
    
    return data, labels, max
    
    

def concat_chr_pos(chr, pos):
    return str(chr)[1:] + str(pos)

def input_fn(data, labels, batch_size, epo):
    
    dataset = tf.data.Dataset.from_tensor_slices((dict(data), labels))
    
    return dataset.repeat(epo).batch(batch_size)

def main(param_path):
    '''
    call this to run. working_dir should be a Path from pathlib.
    '''
    params = loadConfig(param_path)
    input_path = pathlib.Path(params['input_path'])
    working_dir = pathlib.Path(params['directory'])
    del params['input_path']
    del params['directory']
    
    data_dir = input_path / 'nn_data'
    popnet_dir = input_path / 'popnet_small'
    meta_path = input_path / 'meta.csv'
    key_path = input_path / 'filtered_runfile.txt'
    log_path = working_dir / 'nn_log.txt'
    old_logs = working_dir / 'old_logs'
    
    data_path = data_dir / 'data.gz'
    label_path = data_dir / 'label.gz'
    
#     keras = tf.keras
   
    
    if not old_logs.is_dir():
        os.mkdir(old_logs)
    
    if log_path.is_file():
        os.rename(log_path, old_logs / 'nn_log.{0}.txt'.format(int(datetime.timestamp(datetime.now()))))
    
    logger = getLogger('main', log_path)
    
    
    if not os.path.isdir(data_dir):
        logger.info('parsing data..')
        os.mkdir(data_dir)
        newData(popnet_dir, meta_path, key_path, data_path, label_path)
    
    logger.info('using parsed data, starting..')
    
    tf.logging.set_verbosity(tf.logging.INFO)
    
    reps = 1 #how many times do u want to repeat
    for i in range(reps):
        trainAndTest(data_path, label_path, params, working_dir, i)

        

def trainAndTest(data_path, label_path, mod_params, working_dir, i):
    
    i = i + 1
    logger = logging.getLogger('main')
    logger.info('Beginning round {0}'.format(i))
    
    
    data, labels, max = getData(data_path, label_path)
    
        #split to train and test test after shuffling
    epo = 10000
    l = data.shape[1]
    fold = 5
    
    train_size = int(len(data.index) * (fold - 1) / fold)
    
    embedding_size = int(max**0.25) + 1
    
    feature_columns = [tf.feature_column.embedding_column(categorical_column= tf.feature_column.categorical_column_with_identity(key=x, num_buckets=max),
                                                          dimension = embedding_size) for x in data.columns]
    
    #the defaults
    params = {
        'hidden_units': [l*2, l*1, l*0.5, l*0.1],
        'dropout': 0.3,
        'activation': tf.nn.leaky_relu,
        'optimizer': tf.train.ProximalAdagradOptimizer(
                                        learning_rate = 0.03,
                                        l1_regularization_strength= 0.001)}
    
    #changes the ones we want to change from defaults
    for key in mod_params:
        params[key] = mod_params[key]
    
    data = data.sample(frac = 1)
    data, labels = data.align(labels, axis= 0)
    
    train_data = data.iloc[:train_size]
    train_labels = labels.iloc[:train_size]

    test_data = data.iloc[train_size:]
    test_labels = labels.iloc[train_size:]
    
    est = tf.estimator.DNNRegressor(feature_columns = feature_columns,
                                hidden_units= params['hidden_units'],
                                activation_fn = params['activation'],
                                dropout = params['dropout'],
                                optimizer = params['optimizer'],
                                model_dir=working_dir)
     
#     est = tf.estimator.DNNRegressor(feature_columns = feature_columns,
#                             hidden_units= [l*2, l, l*1, l*0.5, l*0.1],
#                             dropout = 0.5,
#                             optimizer = tf.train.ProximalAdagradOptimizer(
#                                         learning_rate = 0.1,
#                                         l1_regularization_strength= 0.001))
    
    
    logger.info('train..')
    est.train(input_fn = lambda : input_fn(train_data, train_labels, train_size, epo), steps = epo)

    
    logger.info('Eval!')
    try:
        res = est.evaluate(input_fn = lambda : input_fn(test_data, test_labels, 1, 1))
    except:
        raise(Exception('still not working lul'))
         
        

    logger.info('round {0}: {1}\nparams used: {2}'.format(i,res,str(params)))

    
if __name__ == '__main__':
    start = time.time()
#     working_dir = pathlib.Path(sys.argv[1])
    param_path = pathlib.Path(sys.argv[1])
    main(param_path)
    end = time.time()
    print('NeuralNet.py completed in {0} seconds'.format(str(end - start)))    
    