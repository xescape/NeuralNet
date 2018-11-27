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

def concat_chr_pos(chr, pos):
    
    return str(chr)[1:] + str(pos)

def input_fn(data, labels, batch_size):
    
    dataset = tf.data.Dataset.from_tensor_slices((dict(data), labels))
    
    return dataset.shuffle(1000).repeat(1).batch(batch_size)

if __name__ == '__main__':
    
    dir = '/data/new/javi/plasmo/new/popnet'
    meta_path = '/data/new/javi/plasmo/new/meta.csv'
    key_path = '/d/data/plasmo/filtered_runfile.txt'
    
    keras = tf.keras

    data, labels, max = id.importData(dir, meta_path, key_path)
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
    

#     msk = data[target].str.contains('not')
#     data = data[~msk]
    
    labels = data.filter(items=[target])
    data = data.filter(items=data.columns.difference([target]))
    
    
    l = data.shape[1]
#     batch_size = data.shape[0] // 10
    
    
    embedding_size = int(l**0.25)
    
    feature_columns = [tf.feature_column.embedding_column(categorical_column= tf.feature_column.categorical_column_with_identity(key=x, num_buckets=max),
                                                          dimension = embedding_size) for x in data.columns] 
    
    print('Training!')
    
    #fake regressor
    est = tf.estimator.DNNRegressor(feature_columns = feature_columns,
                                hidden_units=[2],
                                optimizer = tf.train.ProximalAdagradOptimizer(
                                    learning_rate = 0.1,
                                    l1_regularization_strength= 0.001))
    
    
#     est = tf.estimator.DNNRegressor(feature_columns = feature_columns,
#                                     hidden_units=[l, 1.5 * l, 0.5 * l, 0.1 * l],
#                                     dropout = 0.5,
#                                     optimizer = tf.train.ProximalAdagradOptimizer(
#                                         learning_rate = 0.1,
#                                         l1_regularization_strength= 0.001))
    
    est.train(input_fn = lambda : input_fn(data, labels, batch_size), steps = batch_size * 8)
    eval_result = np.average([est.evaluate(input_fn = lambda : input_fn(data, labels, batch_size * 2), steps = l) for x in range(5)])
    
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    
    

#     y = pp.normalize(y.reshape(1, -1), norm='max').ravel()
    
#     print(y)
    
#     model = tf.keras.Sequential()
#     model.add(keras.layers.Dense(l, activation='relu'))
#     model.add(keras.layers.Flatten())
#     model.add(keras.layers.Dense(int(l * m * 1.5), activation='relu'))
#     model.add(keras.layers.Dense(int(l * 1.5), activation='relu'))
#     model.add(keras.layers.Dense(int(l * 0.3), activation='sigmoid'))
#     model.add(keras.layers.Dense(int(l * 0.2), activation='relu'))
#     model.add(keras.layers.Dense(int(l * 0.5), activation='sigmoid'))
#     model.add(keras.layers.Dense(int(l * 0.5), activation='relu'))
#     model.add(keras.layers.Dense(int(l * 0.5), activation='sigmoid'))
#     model.add(keras.layers.Dense(int(l * 0.5), activation='sigmoid'))
#     model.add(keras.layers.Dense(int(l * 0.5), activation='relu'))
#     model.add(keras.layers.Dense(int(l), activation='softmax'))
    
    
