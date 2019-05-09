'''
Created on Nov 9, 2018

@author: Javi

This file contains the main data import and transformation methods. 
We'll try using pandas

'''
import pandas
import os  
import csv  
import re 
import numpy as np

def formatHex(s):
    '''
    reformats the hex value to something regular
    '''
    pattern = '0x(.+?)$'
    val = re.search(pattern, s.strip()).group(1)
    
    return '#{0:0>6}'.format(val).upper()
    

def importPopNet(dir):
    '''
    imports the data given a popnet result directory. probably will need a
    new method for the stable version
        
    '''

    data_path = os.path.join(dir, 'cytoscape', 'tabNetwork.tsv')
    tab_path = os.path.join(dir, 'persistentMatrix.tab')
    color_path = os.path.join(dir, 'colors.txt')
    
    #create the color dict.
    
#     with open(tab_path) as input:
#         samples = re.split('\n', input.read().strip())
    
    with open(color_path) as input:
        reader = csv.reader(input, delimiter='\t')
        group_colors = {formatHex(x[1]): n for n, x in enumerate(reader)}
        input.seek(0)
        max = sum(1 for line in input)
        
    
    
    with open(data_path) as input:
        d = pandas.read_csv(input, sep='\t')
        for color in group_colors:
            d = d.replace(color, group_colors[color])
    
#     t = d.T.drop(['Chromosome', 'Position'], axis=0)
    
#     print(group_colors)
    
    return d, max
    


def importData(dir, meta_path, key_path):
    '''
    imports the meta data files associated with the
    plasmo data set. will need another one for other 
    data sets with different file structures... 
    
    '''
    
    with open(meta_path) as input:
        meta = pandas.read_csv(input, sep=',', index_col=0, header=0)

        
    with open(key_path) as input:
        key = pandas.read_csv(input, sep='\t', header=0, index_col=13).filter(items=['Run'])

    print(key)
    print(meta)
    
    print('Im trying to merge on a key that doesnt exist')
    
    meta = meta.merge(key, on='Sample_Name')
    
    print('but it was fine anyway')

    
    print(key['Sample_Name'])
    print(meta['Sample_Name'])
    
    meta = meta.set_index('Run')
     
    popnet, max = importPopNet(dir)
    
    return popnet, meta, max



if __name__ == '__main__':
    
    dir = '/d/data/plasmo/popnet'
    meta_path = '/d/data/plasmo/meta.csv'
    key_path = '/d/data/plasmo/filtered_runfile.txt'
    
    importData(dir, meta_path, key_path)