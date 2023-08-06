export TORCH_SEED=0
export GROUP_SEED=0
# python run_genes.py --label FH_AD --chrom 2
python sensitivity_analyses.py

# export TORCH_SEED=0
# export GROUP_SEED=1
# python sensitivity_analyses.py

# export TORCH_SEED=1
# export GROUP_SEED=0
# python sensitivity_analyses.py

# export TORCH_SEED=1
# export GROUP_SEED=1
# python sensitivity_analyses.py