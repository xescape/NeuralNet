#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=12:00:00
#SBATCH --job-name seqprocess
#SBATCH --output=mpi_output_%j.txt
#SBATCH --mail-type=FAIL

module load CCEnv arch/avx512 StdEnv/2018.3
module load python/3.7.0
module load gatk/4.1.2.0
module load samtools/1.9
module load bwa/0.7.17

script_path='/home/j/jparkin/xescape/NeuralNet/Scripts/BamSequenceProcessor.py'

# script_path='/d/workspace/NeuralNet/Scripts/SequenceProcessor.py'
# clear_path='/d/workspace/NeuralNet/Scripts/ClearLock.py'

python3 $script_path scinet
