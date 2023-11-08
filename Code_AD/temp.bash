seeds=(0 73 816 347)
for seed in ${seeds[@]}
do
    export TORCH_SEED=$seed
    time python cov_model.py --label FH_AD > "./Runs/cov_seed"$seed".txt" 2>&1
    time python dummy_genes.py --label FH_AD > "./Runs/dummy_seed"$seed".txt" 2>&1
    time python run_genes.py --label FH_AD --chrom -1 > "./Runs/seed"$seed".txt" 2>&1
done
