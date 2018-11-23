'''
Created on Oct 30, 2018

@author: Javi
'''
import os


if __name__ == '__main__':
    
    dir = ''
    
    os.chdir(dir)
    
    for folder in os.listdir(dir):
        for file in os.listdir(folder)