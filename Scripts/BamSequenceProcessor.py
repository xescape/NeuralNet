'''
Created on Oct 25, 2018

@author: Javi

runs bwa from scratch, I guess, given a folder full of files that include the reference.
This version is just for the natcom data, where we have a bunch of bams files and just run the haplotype caller step, essentially.
various complications from the previous use case have been disabled. modify with care.
'''
import os
from pathlib import Path
import re 
import sys
import subprocess
import multiprocessing as mp
import logging
import shutil

def run(ref, in_path, out_path):
    '''
    takes up TWO threads. Runs everything 
    n is a counter used as the RG so every sample is different
    in_path is the path to the BAM file.
    everything goes into the same log
    '''

    try:
        prefix = in_path.stem
        log_path = out_path / 'log.txt'
        lock_path = out_path / '{0}.lock'.format(prefix)
        is_locked = False


        if lock_path.is_file(): 
            return #someone else already locked it
        else:
            try:
                lock_path.touch()
                is_locked = True
            except:
                return #someone locked at the same time
 
        logger = configLogger(log_path)
        
        if check(log_path, prefix):
            print(prefix + ' already completed.')
            return

        bam_path = in_path
        vcf_path = out_path / '{0}.g.vcf'.format(prefix)

        print('Sample {sample} starting'.format(sample=prefix))
        
        commands = [
            # 'bwa mem {0} {1} {2} > {3}'.format(ref, fqs[0], fqs[1], sam_path),
            # 'samtools view -bt {ref}.fai, -o {bam_path} {sam_path}'.format(ref=ref, bam_path=bam_path, sam_path=sam_path),
            # 'gatk AddOrReplaceReadGroups -I {bam_path} -O {bam_rg_path} -RGID {n} -RGSM {prefix} -RGLB lib{n} -RGPL illumina -RGPU unit{n}'.format(bam_path=bam_path, bam_rg_path=bam_rg_path, n=str(n), prefix=prefix),
            # 'gatk ValidateSamFile -I {bam_rg_path}'.format(bam_rg_path=bam_path),
            # 'samtools sort -o {bam_path} {bam_rg_path}'.format(bam_path=bam_path, bam_rg_path=bam_path),
            'samtools index {bam_path}'.format(bam_path=bam_path),
            'gatk --java-options -Xmx8G HaplotypeCaller -R {ref} -I {bam_path} -O {vcf_path} -ERC GVCF -ploidy 1'.format(ref=ref, bam_path=bam_path, vcf_path=vcf_path)
        ]

        for command in commands:
            try:
                subprocess.run(command, shell=True, check=True)
            except Exception as e:
                print('command error:\n{0}'.format(e))
                return 
                
        logger.info(prefix + ' COMPLETED')
    except Exception as e:
        print(e)
    finally:
        if is_locked and lock_path.is_file():
            lock_path.unlink()#remove lock upon completion


def runRef(ref_path, out_path):
    '''
    preps the reference
    '''
    ref_file = [x for x in ref_path.iterdir() if x.suffix == '.fasta']
    if len(ref_file) > 1:
        raise(Exception('More than 1 fasta in ref folder'))
    else:
        ref_file = ref_file[0]
        
    new_ref_path = out_path / 'ref' / ref_file.name
    new_ref_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ref_file, new_ref_path)
    ref_path = new_ref_path
    ref_dict = ref_path.with_suffix('.dict')

    subprocess.run('bwa index {ref}'.format(ref=ref_path), shell=True)
    subprocess.run('samtools faidx {ref}'.format(ref=ref_path), shell=True)
    if not os.path.isfile(ref_dict):
        subprocess.run('gatk CreateSequenceDictionary -R {ref} -O {ref_dict}'.format(ref=ref_path, ref_dict=ref_dict), shell=True)
    
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

def check(log_path, prefix):
    '''
    path is the log path. Basically checks if the log has that successful part.
    '''
    success_msg = prefix + ' COMPLETED'

    with open(log_path, 'r') as input:
        d = input.read()
    
    if success_msg in d.upper():
        return True
    else:
        return False

def checkRef(log_path):
    '''
    checks to see if the reference is prepped
    '''
    success_msg = 'ref OK'
    try:
        with open(log_path) as input:
            d = input.read()
    except:
        return False
        
    if success_msg in d:
        return True
    else:
        return False

if __name__ == '__main__':
    
    # in_path = Path(sys.argv[1])
    # out_path = Path(sys.argv[2])

    if sys.argv[1] == 'local':
        in_path = Path('/d/data/plasmo/natcom_data/bams')
        out_path = Path('/d/data/plasmo/natcom_data/out')
    else:
        in_path = Path('/scratch/j/jparkin/xescape/plasmo/natcom/bams')
        out_path = Path('/scratch/j/jparkin/xescape/plasmo/nat_out')
    
    os.chdir(out_path)
    log_path = out_path / 'log.txt'
    ref_path = out_path / 'ref' / '3d7.fasta'
    logger = configLogger(log_path)
    
    ref_status = checkRef(log_path)
    if not ref_status:
        runRef(in_path / 'ref', out_path) #that's where the reference is expected to reside before imported.
        logger.info('ref OK')
        
    if not out_path.is_dir():
        out_path.mkdir()
    
    
    samples = [x for x in in_path.iterdir() if x.is_file() and x.name.endswith('bam')]

    print('{0} samples.'.format(len(samples)))

    args = [(ref_path, sample, out_path) for n, sample in enumerate(samples)]
    with mp.Pool(processes=int(mp.cpu_count() // 2)) as pool: #running out of memory
        pool.starmap(run, args)


    print('End of Bam Sequence Processor')

