'''
Created on Nov 6, 2018

@author: Javi
'''

import sys
import os 
import subprocess 


def cat(files):
    
    results = []
    
    for file in files:
        results.append('-V')
        results.append(file)
    
    return results

if __name__ == '__main__':
    
    dir = sys.argv[1]
    ref_path = '/scratch/j/jparkin/xescape/plasmo/3D7.fasta'
#     ref_path = '/d/data/plasmo/bwa/3D7.fasta'
    
    files = [x for x in os.listdir(dir) if x.endswith('.g.vcf')]
    
    os.chdir(dir)
    t = subprocess.check_output(['gatk', 'CombineGVCFs', '-R', ref_path, '-O', 'cohort.g.vcf'] + cat(files))
    print(t)
    print('done!')