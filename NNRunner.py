'''
Created on Jan. 17, 2019

@author: Javi
'''

import pathlib
import os
import subprocess
import re 
import configparser
import sys
import timeit


def getParams():
    '''returns a dict, which contains parameters that we're changing. also includes a name
        edit this one in the future!'''
       
    var = 'relu'
    values = [0.1, 0.15, 0.2, 0.25, 0.3]
    
    
    
    return [{var:val,
             'name':'relu' + str(int(val * 100))} for val in values]

def getHeader(name, out_path, err_path):
    
    h = ["#!/bin/bash", "#SBATCH --nodes=1", "#SBATCH --ntasks-per-node=40", "#SBATCH --time=1:00:00", "#SBATCH --job-name {0}".format(name), \
         "#SBATCH --output={0}".format(out_path), "#SBATCH --error={0}".format(err_path), ""] #extra line at the end
    
    return "\n".join(h)

def getMods():
    
    m = ['module load anaconda3', 'source activate tensorflow', '']
    
    return "\n".join(m)


def makeConfig(params):
    '''create the string for config file'''
    
    c = ['{0}={1}'.format(key, params[key]) for key in params if key != 'name']

    return '\n'.join(c)



def makeScriptAndRun(script_path, root, p, mode):
    '''make the config files as well as the sh script for job submission, then submit it'''
    
    folder = root / p['name']
    config_path = folder / 'config_{0}.txt'.format(p['name'])
    out_path = folder / '{0}_out.txt'.format(p['name'])
    err_path = folder / '{0}_err.txt'.format(p['name'])
    sh_path = folder / 'run_{0}.sh'.format(p['name'])
    
    if not folder.is_dir():
        os.mkdir(folder)
        
    #creates the config file
    #config always needs the root and data_path
    p['directory'] = str(folder)
    p['input_path'] = str(root)
    
    config_str = makeConfig(p)

    with open(config_path, 'w') as config:
        config.write(config_str)    
    

    #creates and submits the job
    if mode == 'local':
        command = "python3 {0} {1}".format(script_path, config_path)
        sh_str = [getHeader(p['name'], out_path, err_path), command]
        with open(sh_path, 'w') as sh:
            sh.write('\n'.join(sh_str))
        subprocess.call(['bash', sh_path])
    elif mode == 'scinet':
        command = "python {0} {1}".format(script_path, config_path)
        sh_str = [getHeader(p['name'], out_path, err_path), getMods(), command]
        with open(sh_path, 'w') as sh:
            sh.write('\n'.join(sh_str))       
        subprocess.call(['sbatch', sh_path])
        
    print('job {0} submitted'.format(p['name']))


def main(script, root, mode):
    '''the stuff'''

    params = getParams()
    
    #we want to make a script for each param set
    for p in params:
        makeScriptAndRun(script, root, p, mode)


if __name__ == '__main__':
    
    mode = sys.argv[1]
    
    if mode == 'local':
        script = '/d/workspace/NeuralNet/NeuralNet.py' #location of the nn script itself
        root = '/d/data/plasmo' #the root working directory. all jobs will be a separate directory under here. 
    elif mode == 'scinet':
        script = '/home/j/jparkin/xescape/NeuralNet/NeuralNet.py'
        root = '/scratch/j/jparkin/xescape/nn'
    else:
        raise(Exception('hey you didnt specify a mode'))
    
    script = pathlib.Path(script)
    root = pathlib.Path(root)

    
    main(script, root, mode)
    