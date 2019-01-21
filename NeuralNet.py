'''
Created on Nov 16, 2018

@author: Javi
'''

import tensorflow as tf
# import keras
# import keras.layers as layers
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
from sklearn import linear_model
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

def linear_error(model, test_set, test_labels):
    '''tries to calculate the absolute error and prediction mean'''
    
    def predict(x):
        return model.predict(x.reshape(1, -1))[0]
       
    average_label = np.mean(test_labels[test_labels.columns[0]])
    
    test_labels['pred'] = model.predict(test_set)
    
    test_labels['loss'] = test_labels.apply(lambda x: abs(x[0] - x[1]), axis=1)
    
    average_pred = np.mean(test_labels['pred'])
    average_loss = np.mean(test_labels['loss'])
    
    print(test_labels)
    return average_label, average_pred, average_loss, test_labels

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
    
    return dataset.shuffle(buffer_size=10000).batch(batch_size, drop_remainder=True).repeat(epo)

def main(param_path):
    '''
    call this to run. working_dir should be a Path from pathlib.
    '''
    tf.enable_eager_execution()
    
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
#         logger.info('parsing data..')
        os.mkdir(data_dir)
        newData(popnet_dir, meta_path, key_path, data_path, label_path)
    
#     logger.info('using parsed data, starting..')
    
    tf.logging.set_verbosity(tf.logging.INFO)
    
    reps = 3 #how many times do u want to repeat
    for i in range(reps):
        res, linear_res = trainAndTest(data_path, label_path, params, working_dir, i)
        res.to_csv(working_dir / 'results_{0}.csv'.format(str(i)))  
        linear_res.to_csv(working_dir / 'results_linear_{0}.csv'.format(str(i))) 

def trainAndTest(data_path, label_path, mod_params, working_dir, i):
    
    def toList(list_str, l):
        return [l * x for x in eval(list_str)]
    
    def exp_decay(init_value, epo):
        return tf.train.exponential_decay(
            learning_rate=init_value,
            global_step=epo,
            decay_steps=1000,
            decay_rate=0.96)
    #set up
    i = i + 1
    logger = logging.getLogger('main')
#     logger.info('Beginning round {0}'.format(i))
    model_dir = working_dir / 'model_{0}'.format(i)
    try:
        os.mkdir(model_dir)
    except:
        pass
    
    #get data from pickle
    data, labels, n_buckets = getData(data_path, label_path)
    
    #set up run parameters
    epo = 10000
    test_epo = 100
    l = data.shape[1]
    fold = 5
    
    train_size = int(len(data.index) * (fold - 1) / fold)
    test_size = int(len(data.index) - train_size)
    
    #we're just going max batch size
    batch_size = int(max(1, train_size/1))
    test_batch_size = int(max(1, test_size/1))
    
    embedding_size = int(n_buckets**0.25) + 1
    
    feature_columns = [tf.feature_column.embedding_column(categorical_column= tf.feature_column.categorical_column_with_identity(key=x, num_buckets=n_buckets),
                                                          dimension = embedding_size) for x in data.columns]
    
    #[4, 2, 1, 0.5, 0.1, 0.05]
    #the defaults
    
    params = {
        'hidden_units': [l*2, l*0.5, l*0.1],
        'dropout': 0.25,
        'activation': 'relu',
        'optimizer': tf.train.AdamOptimizer(learning_rate = exp_decay(0.001, epo))}
    
    #changes the ones we want to change from defaults
    for key in mod_params:
        if key == 'learning_rate':
            params['optimizer'] = tf.train.AdamOptimizer(learning_rate=exp_decay(mod_params['learning_rate'], epo))
        if key == 'hidden_units':
            params['hidden_units'] = toList(mod_params['hidden_units'], l)
        else:
            params[key] = mod_params[key]
    
    #split to train and test test after shuffling
    data = data.sample(frac = 1)
#     data, labels = data.align(labels, axis= 0)
    
    train_data = data.iloc[:train_size]
    train_labels = labels.loc[train_data.index]

    test_data = data.iloc[train_size:]
    test_labels = labels.loc[test_data.index]
    
    
    
    
    est = tf.estimator.DNNRegressor(feature_columns = feature_columns,
                                hidden_units= params['hidden_units'],
                                activation_fn = params['activation'],
                                dropout = params['dropout'],
                                optimizer = params['optimizer'],
                                model_dir=model_dir)
     
#     est = tf.estimator.DNNRegressor(feature_columns = feature_columns,
#                             hidden_units= [l*2, l, l*1, l*0.5, l*0.1],
#                             dropout = 0.5,
#                             optimizer = tf.train.ProximalAdagradOptimizer(
#                                         learning_rate = 0.1,
#                                         l1_regularization_strength= 0.001))
    
    
#     logger.info('Train..')
    est.train(input_fn = lambda : input_fn(train_data, train_labels, batch_size, epo))

    
#     logger.info('Eval!')
    try:
        res = est.evaluate(input_fn = lambda : input_fn(test_data, test_labels, test_batch_size, test_epo))
    except:
        raise(Exception('still not working lul'))
         
#     logger.info('Predict!')    
    
    #Compare against the linear model
    reg = linear_model.Lasso(alpha=0.1)
    reg.fit(train_data, train_labels)
  
    train_score = reg.score(train_data, train_labels)
    test_score = reg.score(test_data, test_labels)

    print('train score = {0} test score = {1}'.format(str(train_score), str(test_score)))
    
    l, p, o, table = linear_error(reg, test_data, test_labels.copy(deep=True))
    
    pred = est.predict(input_fn = lambda : input_fn(test_data, test_labels, 1, 1))
    
    pred_vals = [x['predictions'][0] for x in pred]
    test_labels['pred'] = np.array(pred_vals)
    test_labels['loss'] = test_labels.apply(lambda x: abs(x[0] - x[1]), axis=1)
    
    
    logger.info('round {0}: {1}\nparams used: {2}'.format(i,res,str(params)))
    


    
    return test_labels, table
    
if __name__ == '__main__':
    start = time.time()
#     working_dir = pathlib.Path(sys.argv[1])
    param_path = pathlib.Path(sys.argv[1])
    main(param_path)
    end = time.time()
    print('NeuralNet.py completed in {0} seconds'.format(str(end - start)))    
    
