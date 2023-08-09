# chroms=(3 4 5)
# log_folder="/home/upamanyu/GWANN/Code_AD/NN_Logs/FH_AD_ChrSens7_00_GS10_v4_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam"
# for chrom in ${chroms[@]}
# do
#     export TORCH_SEED=0
#     export GROUP_SEED=0
#     python run_genes.py --label FH_AD --chrom $chrom > "./Runs/chr"$chrom"Run2.txt"
#     chrom_log_folder="/home/upamanyu/GWANN/Code_AD/NN_Logs/Chr"$chrom"Run2_FH_AD_ChrSens7_00_GS10_v4_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam"
#     mv $log_folder $chrom_log_folder
# done

runs=(3 4 5)
chrom=2
log_folder="/home/upamanyu/GWANN/Code_AD/NN_Logs/FH_AD_ChrSens7_00_GS10_v4_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam"
for run in ${runs[@]}
do
    export TORCH_SEED=0
    export GROUP_SEED=0
    python run_genes.py --label FH_AD --chrom $chrom > "./Runs/chr"$chrom"Run"$run".txt"
    chrom_log_folder="/home/upamanyu/GWANN/Code_AD/NN_Logs/Chr"$chrom"Run"$run"_FH_AD_ChrSens7_00_GS10_v4_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam"
    mv $log_folder $chrom_log_folder
done

# export TORCH_SEED=0
# export GROUP_SEED=0
# python sensitivity_analyses.py

# export TORCH_SEED=0
# export GROUP_SEED=1
# python sensitivity_analyses.py

# export TORCH_SEED=1
# export GROUP_SEED=0
# python sensitivity_analyses.py

# export TORCH_SEED=1
# export GROUP_SEED=1
# python sensitivity_analyses.py