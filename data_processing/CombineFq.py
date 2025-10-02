'''
Created on Nov 1, 2018

@author: Javi

single purpose script to combine the fastqs from the plasmodium dataset somehow... using cat

'''

import os
import shutil
import subprocess as sp
import re

def parse(path):
    '''
    parse runaccfiles
    '''
    
    with open(path, 'r') as input:
        data = input.read()
        
    data = re.split('\n', data.strip())[1:]
    
    lines = [re.split('\t', x.strip()) for x in data]
    
    results = [(x[11], x[13], y) for x, y in zip(lines, data)]
    
    return results
    

if __name__ == '__main__':
    
    table_dir = '/home/j/jparkin/xescape'
    
    contig_file = table_dir + '/contigs.txt'
    other_file = table_dir + '/filtered_runfile.txt'
    
    files_dir = '/scratch/j/jparkin/xescape'
    
    contig_path = files_dir + '/contigs'
    other_path = files_dir + '/plasmo'
    
    contigs = parse(contig_file) #contigs should have 3 tuple: the acc, the sample name, and the full line
    other = parse(other_file) #same for this
    
    keys = {}
    sample_name = {x[1]:x[0] for x in other}
    contig_accs = [x[0] for x in contigs]
    
    for x in other:
        if x[0] in contig_accs:
            keys[x[1]] = []
    
    print('\n'.join(['\t'.join([x, y]) for x, y in zip(sorted(keys.keys()), sorted(set([x[1] for x in contigs])))]))
    
    for c in contigs:
        keys[c[1]].append(c)
    
    #move files
    print('moving files')
    move_path = other_path + '/dups'
    if not os.path.isdir(move_path):
        os.mkdir(move_path)
    
    #move the folders
    os.chdir(other_path)
    to_move = [sample_name[x] for x in keys.keys()]
    for folder in to_move:
        shutil.move(folder, move_path)
        os.mkdir(folder)
    
    #cat
    print('starting cats')
    lines = []
    os.chdir(contig_path)
    for name in keys.keys():
        paths1 = ' '.join(['/'.join([x[0], x[0] + "_1.fastq"]) for x in keys[name]])
        paths2 = ' '.join(['/'.join([x[0], x[0] + "_2.fastq"]) for x in keys[name]])
        out_path1 = '/'.join([other_path, sample_name[name], sample_name[name] + "_1.fastq"])
        out_path2 = '/'.join([other_path, sample_name[name], sample_name[name] + "_2.fastq"])
        
        lines = lines + [x[2] for x in keys[name]]
        
        sp.call('cat {0} > {1}'.format(paths1, out_path1), shell = True)
        sp.call('cat {0} > {1}'.format(paths1, out_path2), shell = True)
    
    print('script done zzz')
        
    
    
    
    