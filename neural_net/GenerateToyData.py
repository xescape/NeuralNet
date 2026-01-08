'''
The entry point for the phenotype simulator

we're going to change things up so that we take fake toy data. super fake, like all 0s to start
'''
import sys
import numpy as np
import random
import string
import pandas 
import json

from itertools import starmap
from pathlib import Path

'''
the input file is a linear chromosome painting file. Should be really well formatted, right?
'''
def read_input(in_file):
    df = pandas.read_csv(in_file, sep='\t')
    #we're dropping 3D7 as well.
    df = df.drop(df.columns[:3], axis=1)

    df = df.transpose()    #right now we drop the chr names and positions
    return df

'''
get a blank painting
'''
def generateBlank(n_samples, n_positions):
    chars = string.ascii_uppercase
    names = [x + y + z for x in chars for y in chars for z in chars][:n_samples]

    paintings = np.zeros((n_samples, n_positions))

    return names, paintings

'''
int, float -> list of factors
factors{}
'''
def createFactors(n_factors, p_regulatory):

    def genID():
        letters = string.ascii_lowercase
        length = 5
        return ''.join([random.choice(letters) for i in range(length)])

    def createBaseFactor(n):
        return {'id': genID(), 
                'type': 'base'}
    
    def createRegulatoryFactor(target):
        return {'id': genID(),
                'type': 'reg',
                'target': target['id']}

    n_regulatory = int(np.ceil(n_factors * p_regulatory))
    n_base = int(n_factors - n_regulatory)

    base_factors = list(map(createBaseFactor, np.arange(n_base)))
    reg_factors = list(map(createRegulatoryFactor, [random.choice(base_factors) for x in range(n_regulatory)]))

    return base_factors, reg_factors



'''
paintings is a pandas df. cols are positions. 
'''
def annotateFactors(base_factors, regulatory_factors, paintings):

    def helper(factors, eff_fun):
        for factor in factors:
            pos = next(itr)
            bases = [0,1,2]
            vals = [0] * len(bases)
            vals[1:] = [eff_fun(x) for x in range(len(bases) - 1)]

            factor['pos'] = pos
            factor['alleles'] = bases
            factor['eff_sizes'] = vals

            pos_samples = random.sample(range(n_samples), int(n_samples * percent_allele))
            paintings[pos_samples, pos] = np.random.choice(factor['alleles'][1:], n_samples, replace=True).reshape(n_samples, 1)[pos_samples, 0]

    n_samples, n_positions = paintings.shape
    percent_allele = 0.5

    pos_list = random.sample(range(n_positions), len(base_factors) + len(regulatory_factors))
    itr = iter(pos_list)
    #do base factors first
    base_eff = 5
    #value assignment scheme can be changed
    #current scheme is for the second most common allele to be active and the rest to be inactive.
    helper(base_factors, lambda x: base_eff * random.choice([1, -1]))

    #then regulatory factors
    reg_eff = 2
    helper(regulatory_factors, lambda x: np.float_power(reg_eff, random.choice([1, -1])))
    
    return base_factors, regulatory_factors

def calculatePhenotype(base_factors, regulatory_factors, painting):

    #the base days to clear is 10.
    base_val = 10
    reg_stack = {}
    for reg_factor in regulatory_factors:
        allele = painting[reg_factor['pos']]
        eff = reg_factor['eff_sizes'][reg_factor['alleles'].index(allele)]
        
        #if the target isn't on the stack, init it to 1 and then apply eff
        try:
            reg_stack[reg_factor['target']] += eff 
        except KeyError:
            reg_stack[reg_factor['target']] = 1 + eff

    result = base_val
    for factor in base_factors:
        allele = painting[factor['pos']]
        eff = factor['eff_sizes'][factor['alleles'].index(allele)]

        try:
            result += eff * reg_stack[factor['id']]
        except KeyError:
            result += eff
        
    return result 

def write_factors(base_factors, regulatory_factors, outfile):
    
    with open(outfile, 'w') as output:
        output.write(json.dumps(base_factors + regulatory_factors))

def write_phenotypes(phenotypes, sample_list, outfile):
    #so we're gonna format according to the plasmo metadata
    with open(outfile, 'w') as output:
        #header
        cols = ['name', 'val']
        output.write('{0}\t{1}\n'.format(cols[0], cols[1]))


        for sample, val in zip(sample_list, phenotypes):
            output.write('{0}\t{1}\n'.format(sample, val))

def write_paintings(paintings, out_path):
    '''
    we just use the first two chars as names
    '''
    

    with open(out_path, 'w') as output:
        for name, p in zip(names, paintings):
            output.write(name + '\t' + '\t'.join(p.astype(str)))
            output.write('\n')

if __name__ == '__main__':
    args = sys.argv
    print('start')
    #arguments
    n_samples = 1600
    n_positions = 2300
    n_factors = 32
    p_regulatory = 0.5
    out_path = Path('/mnt/d/data/popnet_paper/newsim')

    painting_outfile = out_path / f'painting_{n_samples}.tsv'
    factor_outfile = out_path / f'factors_{n_samples}.json'
    phenotype_outfile = out_path / f'sim_meta_{n_samples}.tsv'

    #create factors
    base_factors, regulatory_factors = createFactors(n_factors, p_regulatory)

    #annotate factors
    names, chr_paintings = generateBlank(n_samples, n_positions)
    base_factors, regulatory_factors = annotateFactors(base_factors, regulatory_factors, chr_paintings)
    #calculate phenotypes
    phenotypes = starmap(calculatePhenotype, [(base_factors, regulatory_factors, sample) for sample in chr_paintings])
    
    #some sort of output
    write_factors(base_factors, regulatory_factors, factor_outfile)
    write_phenotypes(phenotypes, names, phenotype_outfile)
    write_paintings(chr_paintings, painting_outfile)
    print('finish')