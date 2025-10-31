'''
Created on Nov 21, 2018

@author: Javi
'''

# import tensorflow as tf
import numpy as np
import pandas
import sklearn
from sklearn.preprocessing import OneHotEncoder

def revOneHot(input):
    ohe = OneHotEncoder(categories='auto')
    encoded = ohe.fit_transform(input)
    n_vecs = input.max(axis=0) + 1
    cumulative = np.cumsum(n_vecs) - 1

    res = {}
    for x in range(encoded.shape[1]):
        if x == 0:
            res[x] = 0
            continue
        
        proxy = 1 / (x - cumulative)
        msk = proxy < 0
        proxy = proxy * msk
        res[x] = proxy.argmin()
    
    print(res)



if __name__ == '__main__':

    print(np.argmin(np.array([0, 1, 2, np.nan])))

    data = np.array([[0,0,0,0], [1,1,1,1], [2,2,1,1], [3,3,1,1]])
    revOneHot(data)

