'''
Created on Oct 25, 2018

@author: Javi

runs bwa from scratch, I guess, given a folder full of files that include the reference.
'''
import os
from pathlib import Path
import re 
import sys
import subprocess
import multiprocessing as mp
import logging
import shutil

def run(ref, n, max_step, in_path, out_path):
    '''
    takes up one thread, runs everything from bwa to haplotypecaller
    n is just a counter
    in_path is where the data is. sometimes you can't infer, but from it you can infer the prefix
    '''
    
    def checkStep(log_path):
        '''given the text of a log, return an int representing which step we were on'''
        success_msg = 'step {0}'
        
        try:
            with open(log_path) as input:
                d = input.read()
        except:
            return False
                      
        for x in sorted(range(1, 8), reverse=True):
            if success_msg.format(str(x)) in d:
                return x
        return 0
    
    prefix = in_path.name
    out_path = out_path / prefix
    log_path = out_path / 'log.txt'
    lock_path = out_path / '{0}.lock'.format(os.getpid())

    #lock
    if len(list(out_path.glob('*.lock'))) > 0:
        if not lock_path.is_file():
            return
    else:
        lock_path.touch()

    if len(list(out_path.glob('*.lock'))) != 1:
        lock_path.unlink() #some race condition caused there to be two lock files. give up. 
        return 
    
    #setup and check previous completion
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    
    logger = configLogger(str(n), log_path)
    
    if check(log_path):
        print(prefix + ' already completed.')
        return

    fqs = [x for x in in_path.iterdir() if '.fastq' in x.suffixes]
    sam_path = out_path / '{0}.sam'.format(prefix)
    bam_path = sam_path.with_suffix('.bam')
    bam_rg_path = sam_path.with_suffix('.rg.bam')
    vcf_path = sam_path.with_suffix('.g.vcf')
    
    with open(log_path) as f:
        step = checkStep(f.read())
    
    print(ref)
    
    commands = [
        'bwa mem {0} {1} {2} > {3}'.format(ref, fqs[0], fqs[1], sam_path),
        'samtools view -bt {ref}.fai, -o {bam_path} {sam_path}'.format(ref=ref, bam_path=bam_path, sam_path=sam_path),
        'gatk AddOrReplaceReadGroups -I {bam_path} -O {bam_rg_path} -RGID {n} -RGSM {prefix} -RGLB lib{n} -RGPL illumina -RGPU unit{n}'.format(bam_path=bam_path, bam_rg_path=bam_rg_path, n=str(n), prefix=prefix),
        'gatk ValidateSamFile -I {bam_rg_path}'.format(bam_rg_path=bam_rg_path),
        'samtools sort -o {bam_path} {bam_rg_path}'.format(bam_path=bam_path, bam_rg_path=bam_rg_path),
        'samtools index {bam_path}'.format(bam_path=bam_path),
        'gatk --java-options -Xmx8G HaplotypeCaller -R {ref} -I {bam_path} -O {vcf_path} -ERC GVCF -ploidy 1'.format(ref=ref, bam_path=bam_path, vcf_path=vcf_path)
    ]

    names = ['bwa', 'samtools view', 'Add/ReplaceRG', 'ValidateSam', 'samtools sort', 'samtools index', 'HaplotypeCaller']

    for x in range(step,max_step):
        try:
            subprocess.run(commands[x], shell=True, check=True)
            logger.info('step {0}'.format(x+1))
        except subprocess.CalledProcessError as e:
            logger.error('error encountered during {x}({desc}):\n{cmd}\n{msg}'.format(x=str(x), desc=names[x], cmd=e.cmd, msg=e.stderr))
            lock_path.unlink() #remove lock
            return
    
    lock_path.unlink() #remove lock upon completion


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
    
def configLogger(name, path):
    
    logger = logging.getLogger(name)
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

def check(path):
    '''
    path is the log path. Basically checks if the log has that successful part.
    '''
    success_msg = 'COMPLETED'

    with open(path, 'r') as input:
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
        in_path = Path('/home/javi/seq/plasmo_jz/test')
        out_path = Path('/data/new/javi/plasmo/new/rerun')
    else:
        in_path = Path('/scratch/j/jparkin/xescape/plasmo/plasmo_jz/test')
        out_path = Path('/scratch/j/jparkin/xescape/plasmo/out')
    
    os.chdir(out_path)
    log_path = out_path / 'log.txt'
    ref_path = out_path / 'ref' / '3d7.fasta'
    logger = configLogger('main', log_path)

    TOTAL_STEPS = 7
    run_steps = 7
    
    ref_status = checkRef(log_path)
    if not ref_status:
        runRef(in_path / 'ref', out_path) #that's where the reference is expected to reside before imported.
        logger.info('ref OK')
        
    if not out_path.is_dir():
        out_path.mkdir()
    
    
    samples = [x for x in in_path.iterdir() if x.is_dir() and x.name.startswith('SRR')]
    args = [(ref_path, n, run_steps, sample, out_path) for n, sample in enumerate(samples)]
    with mp.Pool() as pool:
        pool.starmap(run, args)





    #old run process
    # if step < 1:
    #     subprocess.run('bwa mem {0} {1} {2} > {3}'.format(ref, fqs[0], fqs[1], sam_path), shell = True) #runs bwa
    #     logger.info('Step 1')
    # if step < 2: 
    #     subprocess.run(['samtools', 'view', '-bt', ref + '.fai', '-o', bam_path, sam_path])
    #     logger.info('Step 2')
    # if step < 3: 
    #     subprocess.run(['gatk', 'AddOrReplaceReadGroups', '-I', bam_path, '-O', bam_rg_path, '-RGID', str(n), '-RGSM', prefix, '-RGLB', 'lib' + str(n), '-RGPL', 'illumina', '-RGPU', 'unit1'], encoding='utf-8')
    #     logger.info('Step 3')
    # if step < 4: 
    #     try:
    #         subprocess.run(['gatk', 'ValidateSamFile', '-I', bam_rg_path])
    #         logger.info('Step 4')
    #     except subprocess.CalledProcessError:
    #         logger.error('Bam didnt pass QC!')
    #         return
    # if step < 5:
    #     subprocess.run(['samtools', 'sort', '-o', bam_path, bam_rg_path])
    #     logger.info('Step 5')
    # if step < 6:
        
    #     subprocess.run(['samtools', 'index', bam_path])
    #     logger.info('Step 6')
    # if step < 7:
    #     try:
    #         subprocess.run(['gatk', '--java-options', '-Xmx8G', 'HaplotypeCaller', '-R', ref, '-I', bam_path, '-O', vcf_path, '-ERC', 'GVCF', '-ploidy', '1'])
    #         logger.info('Step 7')
    #         logger.info('COMPLETED')    
    #     except subprocess.CalledProcessError:
    #         logger.error('HaplotypeCaller had a problem')
    #         logger.error('Failed!')
