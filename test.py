'''
Created on Nov 21, 2018

@author: Javi
'''

import tensorflow as tf
import numpy as np
import pandas
import sklearn
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':
    cats = ['a', 'b', 'c', 'd', 'e']
    oe = OneHotEncoder(categories = cats, sparse=False)
    data = [[1, 'a','b','c'], [2, 'a', 'b', 'a'], [3,'a','b', 'd']]
    df = pandas.DataFrame(data)
    df.columns = ['A', 'B', 'C', 'D']
     
    df = df.set_index(['A'])
    print(df.columns)

