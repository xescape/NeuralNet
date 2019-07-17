#!/bin/bash 
#SBATCH --nodes=2
#SBATCH --ntasks=80
#SBATCH --time=1:00:00
#SBATCH --job-name mpi_job
#SBATCH --output=mpi_output_%j.txt
#SBATCH --mail-type=FAIL

module load CCEnv arch/avx512 StdEnv/2018.3
module load anaconda3
module load gatk/4.1.2.0
module load samtools/1.9
module load bwa/0.7.17

script_path='/home/j/jparkin/xescape/NeuralNet/Scripts/SequenceProcessor.py'
python $script_path scinet