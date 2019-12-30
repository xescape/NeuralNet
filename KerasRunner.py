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
    
    res = []

    layers = [
        [256, 128, 64, 32, 16],
        [512, 256, 128, 64, 32, 16],
        [64, 128, 128, 32, 16],
        [64, 64, 64],
        [64, 32, 16],
        [64, 16],
        [512, 64],
    ]

    dropout = [0.3, 0.5, 0.8]

    for l in layers:
        for d in dropout:
            res.append((l, d))
  
    return res

def getHeader(name, out_path, err_path):
    
    h = ["#!/bin/bash", "#SBATCH --nodes=1", "#SBATCH --ntasks-per-node=40", "#SBATCH --time=2:00:00", "#SBATCH --job-name {0}".format(name), \
         "#SBATCH --output={0}".format(out_path), "#SBATCH --error={0}".format(err_path), ""] #extra line at the end
    
    return "\n".join(h)

def getMods():
    
    m = ['module load anaconda3/5.1.0', 'source activate tensorflow', '']
    
    return "\n".join(m)


def makeConfig(params):
    '''create the string for config file'''
    
    c = ['{0}={1}'.format(key, params[key]) for key in params if key != 'name']

    return '\n'.join(c)



def makeScriptAndRun(script_path, root, p, mode):
    '''make the config files as well as the sh script for job submission, then submit it
    p[0] is a list of layers, p[1] is the dropout value'''
    print(p)
    name = 'L{0}D{1}'.format('_'.join([str(x) for x in p[0]]), str(p[1]))
    folder = root / name
    out_path = folder / '{0}_out.txt'.format(name)
    err_path = folder / '{0}_err.txt'.format(name)
    sh_path = folder / 'run_{0}.sh'.format(name)
    
    folder.mkdir(parents=True, exist_ok=True)

    #creates and submits the job
    if mode == 'local':
        command = "python3 {script} -i {input} -o {output} -l {layers} -d {dropout} --local".format(script = script_path, input = root, output = folder, layers = ' '.join([str(x) for x in p[0]]), dropout = p[1])
        sh_str = [getHeader(name, out_path, err_path), command]
        with open(sh_path, 'w') as sh:
            sh.write('\n'.join(sh_str))
        subprocess.call(['bash', sh_path])
    elif mode == 'scinet':
        command = "python3 {script} -i {input} -o {output} -l {layers} -d {dropout}".format(script = script_path, input = root, output = folder, layers = ' '.join([str(x) for x in p[0]]), dropout = p[1])
        sh_str = [getHeader(name, out_path, err_path), getMods(), command]
        with open(sh_path, 'w') as sh:
            sh.write('\n'.join(sh_str))       
        subprocess.call(['sbatch', sh_path])
    elif mode == 'scinet-test':
        command = "python3 {script} -i {input} -o {output} -l {layers} -d {dropout}".format(script = script_path, input = root, output = folder, layers = ' '.join([str(x) for x in p[0]]), dropout = p[1])
        sh_str = [getHeader(name, out_path, err_path), getMods(), command]
        with open(sh_path, 'w') as sh:
            sh.write('\n'.join(sh_str))       
        subprocess.call(['bash', sh_path])
        
    print('job {0} submitted'.format(name))


def main(script, root, mode):
    '''the stuff'''

    params = getParams()
    
    #we want to make a script for each param set
    for p in params[:1]:
        makeScriptAndRun(script, root, p, mode)


if __name__ == '__main__':
    
    mode = sys.argv[1]
    
    if mode == 'local':
        script = '/d/workspace/NeuralNet/KerasNet.py' #location of the nn script itself
        root = '/d/data/plasmo/nn_scalar' #the root working directory. all jobs will be a separate directory under here. 
    elif mode == 'scinet' or mode == 'scinet-test':
        script = '/home/j/jparkin/xescape/NeuralNet/KerasNet.py'
        root = '/scratch/j/jparkin/xescape/nn_scalar'
    else:
        raise(Exception('hey you didnt specify a mode'))
    
    script = pathlib.Path(script)
    root = pathlib.Path(root)

    
    main(script, root, mode)
    
