'''
automatic multithread download of pf3k data, and possibly all of malariagen stuff
'''
import logging
import ftplib
import sys
from os import getpid
from time import sleep
from pathlib import Path 
from multiprocessing import Pool, cpu_count
# from subprocess import run, CalledProcessError

def workerInit():
    global ftp
    try:
        ftp = ftplib.FTP('ngs.sanger.ac.uk', 'anonymous', '')
        ftp.cwd('production/pf3k/release_4/BAM')
        print('{0} connected'.format(getpid()))
    except:
        sleep(3)
        print('reconnecting on thread {0}'.format(getpid()))
        workerInit()

def worker(prefix, out_path):
    
    logger = logging.getLogger()
    global ftp
    fn = '{prefix}.bam'.format(prefix=prefix)
    file_path = out_path / fn
    try:
        print('{0} starting sample {1}'.format(getpid(), prefix))
        ftp.retrbinary('RETR {0}'.format(fn), writeFile(file_path), 8192)
        logger.info(prefix)
    except Exception as e:
        print(e)
        sleep(3)
        print('reconnecting on thread {0}'.format(getpid()))
        workerInit()

def writeFile(path):
    def innerWrite(data):
        with open(path, 'wb') as output:
            output.write(data)
    return innerWrite

def loadTable(in_path):
    '''
    input is expected to be a tsv with one line per sample name
    '''
    with open(in_path, 'r') as input:
        d = input.read()
    
    return d.strip('\n').split('\n')
    
def verify(sample_list):
    '''
    given a properly configured ftp object at the desired directory
    checks if all the files are in there
    '''
    try:
        ftp = ftplib.FTP('ngs.sanger.ac.uk', 'anonymous', '')
        ftp.cwd('production/pf3k/release_4/BAM')
        ftp_files = ftp.nlst()
    except:
        sleep(3)
        return verify(sample_list)

    missing = []
    for sample in sample_list:
        if '{0}.bam'.format(sample) not in ftp_files:
            missing.append(sample)
    
    if len(missing) > 0:
        print('The following files are missing: {0}'.format(', '.join(missing)))
        return False 
    else:
        return True

def filter(sample_list, log_path):
    '''
    returns the list of not completed ones
    '''
    try:
        with open(log_path) as input:
            d = input.read()
        
        done = d.strip('\n').split('\n')
        
        not_done = [x for x in sample_list if x not in done]

        return not_done
    except:
        return sample_list


def configLogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s:\n%(message)s\n')
    
    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

def main(in_path, out_path, log_path):
    '''
    main xd
    '''
    n_threads = 4
    sample_list = loadTable(in_path)
    sample_list = filter(sample_list, log_path)

    configLogger(log_path)

    if not verify(sample_list):
        print('terminating due to verification failure')
        return

    with Pool(processes = n_threads, initializer = workerInit) as pool:
        pool.starmap(worker, [(sample, out_path) for sample in sample_list])
    
    print('Main loop complete')
    

if __name__ == '__main__':
    if sys.argv[1] == 'local':
        dir = Path('/d/data/plasmo/natcom_data')
    else:
        dir = Path('/scratch/j/jparkin/xescape/plasmo/natcom')
    in_path = dir / 'test_ids.txt'
    out_path = dir / 'bams'
    log_path = dir / 'log.txt'
    main(in_path, out_path, log_path)
    