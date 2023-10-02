import subprocess
import os
import pandas as pd

logs_folder = '/home/upamanyu/GWANN/Code_AD/NN_Logs'

transfer_cmd = 'rsync -r upamanyu@129.67.155.195:{remote_path} {local_path}.d5'
seeds=[918]
# seeds=[8162, 918, 61, 1502]
for v in ['v4']:
    for gs in [10]:
        for s in seeds:
            exp_logs = f'FH_AD_ChrSens8_{s}{s}_GS{gs}_{v}_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam'
            summ_path = f'{logs_folder}/{exp_logs}/FH_AD_ChrSens8_{s}{s}_GS{gs}_{v}_2500bp_summary.csv'
            if os.path.exists(summ_path):
                cmd = transfer_cmd.format(remote_path=summ_path, local_path=summ_path)
                print(cmd)
                p = subprocess.Popen(cmd, shell=True)
                p.wait()

for v in ['v4']:
    for gs in [10]:
        for s in seeds:
            exp_logs = f'FH_AD_ChrSens8_{s}{s}_GS{gs}_{v}_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam'
            summ_path_d6 = f'{logs_folder}/{exp_logs}/FH_AD_ChrSens8_{s}{s}_GS{gs}_{v}_2500bp_summary.csv'
            summ_path_d5 = f'{logs_folder}/{exp_logs}/FH_AD_ChrSens8_{s}{s}_GS{gs}_{v}_2500bp_summary.csv.d5'
            d5 = pd.read_csv(summ_path_d5)
            d6 = pd.read_csv(summ_path_d6)
            cdf = pd.concat((d5, d6))
            if not os.path.exists(f'{summ_path_d6}.d6'):
                print(summ_path_d6.split('/')[-1])
                os.rename(summ_path_d6, f'{summ_path_d6}.d6')
                cdf.to_csv(summ_path_d6, index=False)
            
# Dummy data
# transfer_cmd = 'rsync -r --ignore-existing upamanyu@129.67.155.195:{remote_path} {local_path}'
# # seeds=[8162, 918, 61, 1502]
# for v in ['v8']:
#     for gs in [10]:
#         for s in seeds:
#             exp_logs = f'FH_AD_ChrDummySens8_{s}{s}_GS{gs}_{v}_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam'
#             dummy_folder = f'{logs_folder}/{exp_logs}'
#             cmd = transfer_cmd.format(remote_path=dummy_folder, local_path=logs_folder)
#             print(cmd)
#             p = subprocess.Popen(cmd, shell=True)
#             p.wait()
#         break
#     break
