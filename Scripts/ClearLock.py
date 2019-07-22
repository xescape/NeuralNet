#Support script for sequence processor, in order to clear the folder locks.
from pathlib import Path
import sys

if __name__ == '__main__':
    if sys.argv[1] == 'local':
        out_path = Path('/d/data/plasmo/test/out')
    else:
        out_path = Path('/scratch/j/jparkin/xescape/plasmo/out')
    
    lock_files = out_path.glob('**/*.lock')

    count = 0
    for f in lock_files:
        count += 1
        f.unlink()
    
    print('{0} lockfiles cleared'.format(count))
    