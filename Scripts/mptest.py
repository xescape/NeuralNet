'''
Created on Oct 29, 2018

@author: Javi
'''
import multiprocessing
import subprocess

def a(x):
    return x

if __name__ == '__main__':
    
    pool = multiprocessing.Pool(100)
    pool.map(a, range(0,1000000))