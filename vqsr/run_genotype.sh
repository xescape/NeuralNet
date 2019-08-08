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

refpath=../ref/3d7.fasta
dbpath=plasmo_db

gatk --java-options "-Xmx8g" GenotypeGVCFs -R $refpath -V gendb://$dbpath -O combined.vcf.gz --tmp-dir=./tmp
