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

refpath=/scratch/j/jparkin/xescape/plasmo/out/ref/3d7.fasta
train1=train1.vcf.gz
train2=train2.vcf.gz
train3=train3.vcf.gz
input=combined.vcf.gz

dbpath=plasmo_db

gatk VariantRecalibrator \
   -R $refpath \
   -V $input \
   --resource:train1,known=false,training=true,truth=true,prior=15.0 train1.vcf.gz \
   --resource:train2,known=false,training=true,truth=false,prior=12.0 train2.vcf.gz \
   --resource:train3,known=false,training=true,truth=false,prior=10.0 train3.vcf.gz \
   --resource:combined_filtered,known=true,training=false,truth=false,prior=2.0 combined_filtered.vcf.gz \
   -an QD -an MQ -an MQRankSum -an ReadPosRankSum -an FS -an SOR \
   -mode SNP \
   -O output.recal \
   --tranches-file output.tranches \
   --rscript-file output.plots.R
