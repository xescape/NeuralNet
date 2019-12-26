#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=12:00:00
#SBATCH --job-name seqprocess
#SBATCH --output=mpi_output_%j.txt
#SBATCH --mail-type=FAIL

module load NiaEnv
module load anaconda3/2018.12
module load gcc/7.3.0
module load mcl/14-137

popnetpath="/home/j/jparkin/xescape/PopNetNN/PopNet.py"
configpath="/scratch/j/jparkin/xescape/plasmo/nat_popnet/nat_scinet_config.txt"

python3 $popnetpath $configpath



