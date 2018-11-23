'''
Created on Nov 21, 2018

@author: Javi
'''

import tensorflow as tf
import numpy as np
import pandas

def f(a, b, c):
    
    return tf.one_hot(a, c), b

if __name__ == '__main__':
    
    a = pandas.DataFrame({'a':[1,2], 'b':[3,4]})
    a.index = ['x', 'y']
    
    b = pandas.DataFrame({'a':[5], 'b':[6], 'c':[7]})
    b.index = ['z']

    
    c = a.T.join(b.T)
    print(c)
    print(c.max().max())
#     print(dataset)
#     dataset = tf.data.Dataset.from_tensor_slices((a, b))
#     dataset = dataset.map(lambda x, y: tuple(tf.py_func(f, [x, y, c], [tf.int32, b.dtype])))
     
#     with tf.Session() as sess:
#         itr = dataset.make_one_shot_iterator()
#         print(sess.run(itr.get_next()))
#         print(sess.run(itr.get_next()))