'''
Created on Oct 25, 2018

@author: Javi

runs bwa from scratch, I guess, given a folder full of files that include the reference.
'''
import os
import re 
import sys
import subprocess
import multiprocessing as mp
import logging
import shutil

def run_wrapper(args):
    return run(*args)

def run(ref, prefix, n, out_path):
    '''
    takes up one thread, runs everything from bwa to haplotypecaller
    n is just a counter
    '''
    
    def checkStep(text):
        '''given the text of a log, return an int representing which step we were on'''
        for x in sorted(range(1, 8), reverse=True):
            if re.search(str(x), text):
                return x
        return 0
    
    print('worker starting sample ' + prefix)
    
#     prefix = re.match('^(.+?)[.]fastq$', prefix).group(1)
    out_path = out_path + '/' + prefix
    log_path = out_path + '/log.txt'
    
    if check(log_path):
        print(prefix + ' already completed.')
        return
        
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    
    fqs = ['/'.join(['.', prefix, x]) for x in os.listdir('./'+prefix) if x.endswith('.fastq')]
    sam_path = '{0}/{1}.sam'.format(out_path, prefix)
    bam_path = '{0}/{1}.bam'.format(out_path, prefix)
    bam_rg_path = '{0}/{1}_rg.bam'.format(out_path, prefix)
    vcf_path = '{0}/{1}.g.vcf'.format(out_path, prefix)
    
    with open(log_path) as f:
        step = checkStep(f.read())
    
    logger = getLogger(prefix, log_path)
    
    if step < 1:
        t = subprocess.check_output('bwa mem {0} {1} {2} > {3}'.format(ref, fqs[0], fqs[1], sam_path), shell = True, encoding='utf-8') #runs bwa
        logger.info('Step 1 complete')
    if step < 2: 
        t = subprocess.check_output(['samtools', 'view', '-bt', ref + '.fai', '-o', bam_path, sam_path], encoding='utf-8')
        logger.info('Step 2 complete')
    if step < 3: 
        t = subprocess.check_output(['gatk', 'AddOrReplaceReadGroups', '-I', bam_path, '-O', bam_rg_path, '-RGID', str(n), '-RGSM', prefix, '-RGLB', 'lib' + str(n), '-RGPL', 'illumina', '-RGPU', 'unit1'], encoding='utf-8')
        logger.info('Step 3 complete')
    if step < 4: 
        try:
            t = subprocess.check_output(['gatk', 'ValidateSamFile', '-I', bam_rg_path], encoding='utf-8')
            logger.info('Step 4 complete')
        except subprocess.CalledProcessError:
            logger.error('Bam didnt pass QC!')
            return
    if step < 5: 
        t = subprocess.check_output(['samtools', 'sort', '-o', bam_path, bam_rg_path], encoding='utf-8')
        logger.info('Step 5 complete')
    if step < 6: 
        t = subprocess.check_output(['samtools', 'index', bam_path], encoding='utf-8')
        logger.info('Step 6 complete')
    if step < 7:
        try:
            t = subprocess.check_output(['gatk', '--java-options', '-Xmx8G', 'HaplotypeCaller', '-R', ref, '-I', bam_path, '-O', vcf_path, '-ERC', 'GVCF', '-ploidy', '1'], encoding='utf-8')
            logger.info('Step 7 complete')
            logger.info('Success!')    
        except subprocess.CalledProcessError:
            logger.error('HaplotypeCaller had a problem')
            logger.error('Failed!')
            return    
    

def getLogger(name, path):
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: \n %(message)s \n')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

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
    out_path = 'out'
    log_path = out_path + '/log.txt'
    
    
    ref = '3D7.fasta'
    ref_dict = '3D7.dict'
    
    os.chdir(dir)
    
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    logger = getLogger('main', log_path)
    
    t = subprocess.check_output(['bwa', 'index', ref], encoding='utf-8')
    logger.info(t)
      
    t = subprocess.check_output(['samtools', 'faidx', ref], encoding='utf-8')
    logger.info(t)
    
    if not os.path.isfile(ref_dict):
        t = subprocess.check_output(['gatk', 'CreateSequenceDictionary', '-R', ref, '-O', ref_dict], encoding='utf-8')
        logger.info(t)
    

    args = [(ref, x, n, out_path) for n, x in enumerate(os.listdir(dir)) if (os.path.isdir(x) and x.startswith('SRR'))]
    pool = mp.Pool(40)
    pool.map(run_wrapper, args)
