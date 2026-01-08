'''
Created on Nov 2, 2018

@author: Javi
'''

import os
import sys
import re   

def check(path):
    '''
    path is the log path. Basically checks if the log has that successful part.
    '''
    
    with open(path, 'r') as input:
        data = input.read()
    
    if re.search('Success!', data):
        return True
    return False


if __name__ == '__main__':
    dir = sys.argv[1]
    outpath = os.path.join(dir, 'incomplete.txt')
    
    done = []
    not_done = []
    
    for folder in [x for x in os.listdir(dir) if (x.startswith('SRR'))]:
        log_path = os.path.join(dir, folder, 'log.txt')
        if check(log_path):
            done.append(folder)
        else: not_done.append(folder)
    
    print('{0} done out of {1}'.format(len(done), len(done + not_done)))
    
    
    