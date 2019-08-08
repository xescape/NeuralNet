'''
makes a map for all the files in a directory ending in something
'''

from pathlib import Path 
import os
import sys 

if __name__ == "__main__":

    site = sys.argv[1]

    if site == 'local':
        folder = '/d/data/plasmo/natcom_data/bams'
        output = '/d/data/plasmo/natcom_data/bams/sample_map.txt'
    else:
        folder = '/scratch/j/jparkin/xescape/plasmo/out/vcfs' #folder name
        output = '/scratch/j/jparkin/xescape/plasmo/out/vcfs/sample_map.txt' #file name
    
    
    in_path = Path(folder)
    out_path = Path(output)

    suffix = '.vcf'
    
    samples = [f for f in in_path.iterdir() if f.is_file() and f.suffix == suffix]
    print('{0} samples found'.format(len(samples)))

    with open(out_path, 'w') as out:
        for f in samples:
            out.write("{sample}\t{file}\n".format(sample=f.with_suffix('').stem, file=f.name))
            
