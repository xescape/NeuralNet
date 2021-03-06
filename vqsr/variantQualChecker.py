'''
Created on Jan. 9, 2019

@author: Javi
'''
import sys 
import re 
import os 

if __name__ == '__main__':
    
    dir = '/d/data/plasmo'
    
    path = dir + '/sub.tsv'
    out_path = dir + '/test.tsv'
    
    
    with open(path, 'r') as input, open(out_path, 'w') as output:
        
        output.write(input.readline())
        
        for line in input:
            elements = re.split('\t', line.strip())
            ref = elements[2]
            elements = [x if x != '.' else ref for x in elements ]
            
            if len(set(elements[2:])) != 1:
                output.write('\t'.join(elements) + '\n')
    
    print('done!')