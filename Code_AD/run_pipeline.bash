# seeds=(281 363)
# group_size=10
# for seed in ${seeds[@]}
# do
#     export TORCH_SEED=$seed
#     export GROUP_SEED=$seed
#     export GROUP_SIZE=$group_size
#     time python dummy_genes.py --label FH_AD > "./Runs/dummy"$seed".txt" 2>&1
# done

# seeds=(0 4250 937 89 172 37 363 281 142 7282)
seed=7282
# chroms=(21 19 17 15 13 11 9 7 5 3 1)
chroms=(22 20 18 16 14 12 10 8 6 4 2)
group_size=10
exp_name="Sens8_"$seed""$seed"_GS"$group_size"_v4"
log_folder="/home/upamanyu/GWANN/Code_AD/NN_Logs/FH_AD_Chr"$exp_name"_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam"

# export TORCH_SEED=$seed
# export GROUP_SEED=$seed
# export GROUP_SIZE=$group_size
# time python cov_model.py --label FH_AD > "./Runs/cov"$seed".txt" 2>&1

for chrom in ${chroms[@]}
do
    chrom_log_folder="/home/upamanyu/GWANN/Code_AD/NN_Logs/Chr"$chrom"_FH_AD_Chr"$exp_name"_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam"
    # mv $chrom_log_folder $log_folder
    export TORCH_SEED=$seed
    export GROUP_SEED=$seed
    export GROUP_SIZE=$group_size
    time python run_genes.py --label FH_AD --chrom $chrom > "./Runs/chr"$chrom"_"$seed".txt" 2>&1
    mv $log_folder $chrom_log_folder
done

# runs=(3 4 5)
# chrom=2
# log_folder="/home/upamanyu/GWANN/Code_AD/NN_Logs/FH_AD_ChrSens7_00_GS10_v4_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam"
# for run in ${runs[@]}
# do
#     export TORCH_SEED=0
#     export GROUP_SEED=0
#     python run_genes.py --label FH_AD --chrom $chrom > "./Runs/chr"$chrom"Run"$run".txt"
#     chrom_log_folder="/home/upamanyu/GWANN/Code_AD/NN_Logs/Chr"$chrom"Run"$run"_FH_AD_ChrSens7_00_GS10_v4_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam"
#     mv $log_folder $chrom_log_folder
# done

# runs=(1 2)
# log_folder="/home/upamanyu/GWANN/Code_AD/NN_Logs/FH_AD_ChrSens8_00_GS10_v4_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam"
# for run in ${runs[@]}
# do
#     export TORCH_SEED=0
#     export GROUP_SEED=0
#     python sensitivity_analyses.py > "./Runs/Run"$run".txt"
#     new_log_folder="/home/upamanyu/GWANN/Code_AD/NN_Logs/Run"$run"_FH_AD_ChrSens8_00_GS10_v4_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam"
#     mv $log_folder $new_log_folder
# done

# group_sizes=(10)
# seeds=(0 712 163 4250 8162 918 61 1502 362)
# seeds=(937 89 172)
# seeds=(937)
# for group_size in ${group_sizes[@]}
# do
#     for seed in ${seeds[@]}
#     do
#         export TORCH_SEED=$seed
#         export GROUP_SEED=$seed
#         export GROUP_SIZE=$group_size
#         python sensitivity_analyses.py
#     done
# done