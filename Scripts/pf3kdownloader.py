'''
automatic multithread download of pf3k data, and possibly all of malariagen stuff
'''
import logging
from pathlib import Path 
from subprocess import run, CalledProcessError

def worker(prefix, out_path):
    logger = logging.getLogger()
    url = Path('ftp://ngs.sanger.ac.uk/production/pf3k/release_4/BAM/') / '{prefix}.bam'.format(prefix=prefix)
    try:
        run('wget -P {out_path} {url}'.format(out_path=out_path, url=url), shell = True, check = True)
        logging.info('{0} OK'.format(prefix))
    except CalledProcessError as e:
        logging.info('{prefix} FAIL:\n{cmd}\n{msg}'.format(prefix=prefix, cmd=e.cmd, msg=e.stderr))

def check(prefixes, log_path):
    '''
    returns the list of not completed ones
    '''
    


if __name__ == '__main__':

    input = Path()
    output = Path()
