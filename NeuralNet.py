'''
Created on Nov 16, 2018

@author: Javi
'''

import tensorflow as tf
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import ImportData as id
import numpy as np
import pandas
import sys
import os
import logging

def op_model(features, labels, mode):
    return

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

if __name__ == '__main__':
    
    working_dir = '/d/data/plasmo'
    data_dir = working_dir + '/nn_data'
    popnet_dir = working_dir + '/popnet_small'
    meta_path = working_dir + '/meta.csv'
    key_path = working_dir + '/filtered_runfile.txt'
    
    data_path = data_dir + '/data.gz'
    label_path = data_dir + '/label.gz'
    
    keras = tf.keras
    
    logger = getLogger('main', working_dir + '/nn_log.txt')
    
    
    if not os.path.isdir(data_dir):
        logger.info('getting new data')
        os.mkdir(data_dir)
        newData(popnet_dir, meta_path, key_path, data_path, label_path)
    
    logger.info('recovering data')
    data, labels, max = getData(data_path, label_path)
    
    #split to train and test test after shuffling
    epo = 1000
    l = data.shape[1]
    fold = 5
    
    train_size = int(len(data.index) * (fold - 1) / fold)
    embedding_size = int(max**0.25) + 1
    
    feature_columns = [tf.feature_column.embedding_column(categorical_column= tf.feature_column.categorical_column_with_identity(key=x, num_buckets=max),
                                                          dimension = embedding_size) for x in data.columns] 
    
    
    tf.logging.set_verbosity(tf.logging.INFO)
    eval_result = []
    
    params = {
        'hidden_units': [l*2, l, l*1, l*0.5, l*0.1],
        'dropout': 0.5,
        'optimizer': tf.train.ProximalAdagradOptimizer(
                                        learning_rate = 0.1,
                                        l1_regularization_strength= 0.001)}
    
    
    for x in range(5):
        logger.info('Beginning round {0}'.format(x + 1))
        
        data = data.sample(frac = 1)
        data, labels = data.align(labels, axis= 0)
        
        train_data = data.iloc[:train_size]
        train_labels = labels.iloc[:train_size]
       
        test_data = data.iloc[train_size:]
        test_labels = labels.iloc[train_size:]
        
        est = tf.estimator.DNNRegressor(feature_columns = feature_columns,
                                    hidden_units= params['hidden_units'],
                                    dropout = params['dropout'],
                                    optimizer = params['optimizer'])
         
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
            eval_result.append(res)
        except:
            raise(Exception('still not working lul'))
        
        logger.info('round {0} end with results {1}'.format(x + 1, res))
 
    for i, x in enumerate(eval_result):
        logger.info('round {0}: {1}'.format(i + 1,x))
        
    logger.info('params used: {0}'.format(str(params)))

    
    
