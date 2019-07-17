#Support script for sequence processor, in order to clear the folder locks.
from pathlib import Path
import sys

if __name__ == '__main__':

    out_path = Path(sys.argv[1])
    lock_files = out_path.glob('**/*.lock')

    for f in lock_files:
        f.unlink()
    