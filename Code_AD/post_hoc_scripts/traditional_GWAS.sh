#!/bin/bash

# Define the output file
pgen_base=/mnt/sdh/upamanyu/GWANN/GWANN_pgen
base_folder=/mnt/sdh/upamanyu/GWANN/GWAS
output_file=${base_folder}/FH_AD_summary_stats.txt

# Loop over the chromosomes
for i in {1..22}
do
    # Run PLINK2 logistic regression
    plink2 \
        --pfile ${pgen_base}/UKB_chr${i} \
        --keep ${base_folder}/keep.txt \
        --extract ${base_folder}/extract.txt \
        --covar ${base_folder}/covariates.txt \
        --pheno ${base_folder}/pheno.txt \
        --pheno-name FH_AD \
        --logistic \
        --threads 40 \
        --out ${base_folder}/chr${i}_output

    # If this is the first chromosome, copy the header to the output file
    if [ $i -eq 1 ]
    then
        head -n 1 ${base_folder}/chr${i}_output.FH_AD.glm.logistic.hybrid > $output_file
    fi

    # Append the results to the output file, skipping the header
    tail -n +2 ${base_folder}/chr${i}_output.FH_AD.glm.logistic.hybrid >> $output_file
done