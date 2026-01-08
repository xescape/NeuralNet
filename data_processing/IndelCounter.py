'''
Created on Feb. 18, 2019

@author: Javi
'''
from pathlib import Path
import numpy as np
import re
import sys

def count(file):
    '''
    the first one is indels, the second one is not indels
    '''
    
    
    def parseline(line):
        elements = re.split('\t', line)
        ref = elements[3]
        non_ref = elements[4]
        
        for e in (ref, non_ref):
            sub_elements = re.split(',', e)
            for sub_e in sub_elements:
                if len(sub_e) > 1:
                    if sub_e != '<NON_REF>':
                        return True
        return False
        
    
    res = np.zeros(2)
    
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            else:
                if parseline(line):
                    res[0] += 1
                else:
                    res[1] += 1
    return res

def main(folder):
    
    folder_path = Path(folder)
    
    res = np.zeros(2)
    
    for file in folder_path.iterdir():
        if file.name.endswith('.g.vcf'):
            res = res + count(file)
    
    print(res)

if __name__ == '__main__':
    
    mode = sys.argv[1]
    
    if mode == 'local':
        dir = '/d/data/plasmo/gvcf/test'
    elif mode == 'scinet':
        dir = '/scratch/j/jparkin/xescape/plasmo/out/gvcf'
    else:
        raise Exception('no mode')
    
    main(dir)