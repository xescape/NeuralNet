'''
automatic multithread download of pf3k data, and possibly all of malariagen stuff
'''
import logging
import ftplib
import sys
import re
from os import getpid, chdir
from time import sleep
from pathlib import Path 
from multiprocessing import Pool, cpu_count
from subprocess import run, CalledProcessError

def worker(prefix, out_path):
    
    logger = logging.getLogger()
    # fn = '{prefix}.bam'.format(prefix=prefix)
    chdir(out_path)
    try:
        print('{0} starting sample {1}'.format(getpid(), prefix))
        run('wget -nd -c -t 10 --random-wait ftp://ngs.sanger.ac.uk/production/pf3k/release_4/BAM/{0}.bam'.format(prefix), shell=True, check=True)
        logger.info(prefix)
    except CalledProcessError as e:
        print(e)


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
    
def verify(sample_list, missing_path):
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
        return verify(sample_list, missing_path)

    missing = []
    for sample in sample_list:
        if '{0}.bam'.format(sample) not in ftp_files:
            missing.append(sample)
    
    if len(missing) > 0:
        print('The following files are missing: {0}'.format(', '.join(missing)))
        with open(missing_path, 'w') as output:
            output.write('\n'.join(missing))
        return [sample for sample in sample_list if sample not in missing] 
    else:
        return sample_list

def filter(sample_list, log_path):
    '''
    returns the list of not completed ones
    '''
    try:
        with open(log_path) as input:
            d = input.read()
        
        pat = re.compile('(?=:\n).+')
        done = re.findall(pat, d)
        
        not_done = [x for x in sample_list if x not in done]
        print('{0} sampled already completed. {1} to go.'.format(len(done), len(not_done)))

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

def main(in_path, out_path, log_path, missing_path):
    '''
    main xd
    '''
    n_threads = 8
    sample_list = loadTable(in_path)
    sample_list = filter(sample_list, log_path)

    configLogger(log_path)

    sample_list = verify(sample_list, missing_path)

    with Pool(processes = n_threads) as pool:
        pool.starmap(worker, [(sample, out_path) for sample in sample_list])
    
    print('Main loop complete')
    

if __name__ == '__main__':
    if sys.argv[1] == 'local':
        dir = Path('/d/data/plasmo/natcom_data')
    else:
        dir = Path('/scratch/j/jparkin/xescape/plasmo/natcom')
    in_path = dir / 'ids.txt'
    out_path = dir / 'bams'
    log_path = dir / 'log.txt'
    missing_path = dir / 'missing.txt'
    main(in_path, out_path, log_path, missing_path)
    