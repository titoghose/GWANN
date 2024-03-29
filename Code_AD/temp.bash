seeds=(0 73 347 816)
group_size=20
freeze_cov=(1)
for fc in ${freeze_cov[@]}
do
    for seed in ${seeds[@]}
    do
        export TORCH_SEED=$seed
        export GROUP_SEED=$seed
        export GROUP_SIZE=$group_size
        export FREEZE_COV=$fc
        time python cov_model.py --label FH_AD > "./Runs/"$seed"_chrom.txt" 2>&1
        time python run_genes.py --label FH_AD --chrom -1 > "./Runs/"$seed"_genes.txt" 2>&1
        time python dummy_genes.py --label FH_AD > "./Runs/"$seed"_dummy.txt" 2>&1
    done
done

# seeds=(937 89 172 363 37 163)
# seed=281
# chroms=(21 19 17 15 13 11 9 7 5 3 1)
# group_size=10
# exp_name="Sens8_"$seed""$seed"_GS"$group_size"_v4"
# log_folder="/home/upamanyu/GWANN/Code_AD/NN_Logs/FH_AD_Chr"$exp_name"_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam"
# for chrom in ${chroms[@]}
# do
#     chrom_log_folder="/home/upamanyu/GWANN/Code_AD/NN_Logs/Chr"$chrom"_FH_AD_Chr"$exp_name"_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam"
#     # mv $chrom_log_folder $log_folder
#     export TORCH_SEED=$seed
#     export GROUP_SEED=$seed
#     export GROUP_SIZE=$group_size
#     time python run_genes.py --label FH_AD --chrom $chrom > "./Runs/chr"$chrom".txt" 2>&1
#     mv $log_folder $chrom_log_folder
# done
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
