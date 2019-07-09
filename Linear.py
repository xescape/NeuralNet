'''
Created on Nov 16, 2018

@author: Javi
'''

# import tensorflow as tf
# import keras
# import keras.layers as layers
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import ImportData as id
import numpy as np
import pandas
import sys
import os
import pathlib
import time
import re
import math
from datetime import datetime

from sklearn import linear_model
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
# 
# def loadConfig(config_path):
#     '''simple load config. only get lines that are like ^X=Y\n'''
#     
#     line_pattern = '^[^#]+?[=].+?'
#     
#     res = {}
#     with open(config_path, 'r') as f:
#         for line in f:
#             line = line.rstrip('\n')
#             if re.match(line_pattern, line):
#                 split = line.split('=')
#                 try:
#                     res[split[0].strip()] = float(split[1].strip())
#                 except:
#                     res[split[0].strip()] = split[1].strip()
# 
#     return res
    
    
# def getLogger(name, path):
#     
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.INFO)
#     
#     fh = logging.FileHandler(path)
#     fh.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: \n %(message)s \n')
#     fh.setFormatter(formatter)
#     logger.addHandler(fh)
#     
#     sh = logging.StreamHandler()
#     sh.setLevel(logging.INFO)
#     logger.addHandler(sh)
#     
#     return logger
#     
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



def abs_error(model, test_set, test_labels):
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
def main():
    '''
    call this to run. working_dir should be a Path from pathlib.
    '''
    
    input_path = pathlib.Path('/d/data/plasmo/')
    working_dir = pathlib.Path('/d/data/plasmo/linear')
    
    data_dir = input_path / 'nn_data'
    popnet_dir = input_path / 'popnet_small'
    meta_path = input_path / 'meta.csv'
    key_path = input_path / 'filtered_runfile.txt'
    log_path = working_dir / 'nn_log.txt'
    old_logs = working_dir / 'old_logs'
    
    data_path = data_dir / 'data.gz'
    label_path = data_dir / 'label.gz'
    
#     keras = tf.keras
   
    if not working_dir.is_dir():
        os.mkdir(working_dir)
        
    if not old_logs.is_dir():
        os.mkdir(old_logs)
    
    if log_path.is_file():
        os.rename(log_path, old_logs / 'nn_log.{0}.txt'.format(int(datetime.timestamp(datetime.now()))))
    
#     logger = getLogger('main', log_path)
    
    
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
        newData(popnet_dir, meta_path, key_path, data_path, label_path)
    
    
#     tf.logging.set_verbosity(tf.logging.INFO)
    
    reps = 2 #how many times do u want to repeat
    for i in range(reps):
        res = trainAndTest(data_path, label_path, i)
        res.to_csv(working_dir / 'linear_result2.csv')

        

def trainAndTest(data_path, label_path, i):

    data, labels, max = getData(data_path, label_path)

    fold = 5
    train_size = int(len(data.index) * (fold - 1) / fold)

    data = data.sample(frac = 1)
#     labels, data = labels.align(data, axis= 0)
    data = pandas.get_dummies(data, columns=data.columns)
    
    train_data = data.iloc[:train_size]
    train_labels = labels.loc[train_data.index]

    test_data = data.iloc[train_size:]
    test_labels = labels.loc[test_data.index]
    
    
    reg = linear_model.Lasso(alpha=0.1)
    reg.fit(train_data, train_labels)
  
    train_score = reg.score(train_data, train_labels)
    test_score = reg.score(test_data, test_labels)

    print('train score = {0} test score = {1}'.format(str(train_score), str(test_score)))
    
    l, p, o, table = abs_error(reg, test_data, test_labels)
    
    print('results:')
    print('label average ' + str(l))
    print('pred average ' + str(p))
    print('loss average ' + str(o))
    
    return table
    
    
if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('Linear.py completed in {0} seconds'.format(str(end - start)))    
    