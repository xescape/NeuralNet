'''
Javi
Runs various GATK steps locally
Part of the NN project
'''

from subprocess import run 

def runCommand(args):
    '''
    take a list of args and run them as a gatk command
    assumes the args have the dash, and the first one is the tool name
    '''

    arg_list = []
    for arg in args:
        arg_list.extend([arg[0], arg[1]])
    
    run(arg_list)

def selectVariantsSNP(reference, input_path, output_path):
    snp_args = [
        ('gatk', 'SelectVariants'),
        ('-R', reference),
        ('-V', input_path),
        ('--select-type', 'SNP'),
        ('-o', output_path)
    ]

    runCommand(snp_args)

def main():
'''
defines the list of commands to read. pass in any inputs
'''

