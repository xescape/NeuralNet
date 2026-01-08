'''
Created on Oct 24, 2018

@author: Javi
'''
import os
import re 
import subprocess
import multiprocessing as mp
from pathlib import Path

def load(file_path):
    
    with open(file_path, 'r') as input:
        data = input.read()
        
    
    data = re.split('\n', data.strip('\n'))
    
    return data

def run(exe_path, srr, out_path):
    subprocess.run([exe_path, srr, '-O', out_path], stdout = subprocess.PIPE)
    
    

if __name__ == '__main__':
    
    exe_path = Path()
    srr_path = r'D:\Documents\data\plasmo\new_accs.txt'
    out_path = r'Z:\plasmo_jz'
    
    os.chdir(r'D:')
    
    srr_list = load(srr_path)
    
    for srr in srr_list:
        run(exe_path, srr, out_path)
        print('downloaded ' + srr + '\n')
    
    