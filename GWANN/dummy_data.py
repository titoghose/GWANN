import os
import time
import subprocess
import pandas as pd

def run_cmd(cmd:str) -> None:
    subprocess.run(cmd, shell=True)

def dummy_plink(samples:list, num_snps:int, dosage_freq:float, 
                out_folder:str, file_prefix:str='') -> None:
    num_samples = len(samples)
    if file_prefix == '':
        file_prefix = 'Dummy'
    out_prefix = f'{out_folder}/{file_prefix}_{num_snps}snps_{dosage_freq:.2f}dosfreq'
    if os.path.exists(f'{out_prefix}.pgen'):
        return out_prefix
    
    
    cmd = f'plink2 --dummy {num_samples} {num_snps} acgt'
    cmd += f' dosage-freq={dosage_freq}'
    cmd += f' --out {out_prefix}'
    cmd += f' --make-pgen'
    time.sleep(1)
    run_cmd(cmd=cmd)
    modify_psam(out_prefix, samples)
    
    return out_prefix

def modify_psam(file_prefix:str, samples:list) -> None:
    df = pd.read_csv(f'{file_prefix}.psam', sep='\t')
    df['#IID'] = samples
    df.to_csv(f'{file_prefix}.psam', sep='\t', index=False)