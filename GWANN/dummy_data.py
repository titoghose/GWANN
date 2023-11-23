import os
import time
import subprocess
import pandas as pd

def run_cmd(cmd:str) -> None:
    subprocess.run(cmd, shell=True)

def dummy_plink(samples:list, num_snps:int, dosage_freq:float,
                out_folder:str, var_pos_offset:int=0, 
                file_prefix:str='') -> None:
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
    
    # Random seed is set using the seconds in system time, so it
    # is important to sleep for 1 second to avoid two datasets
    # being generated with the same random seed.
    time.sleep(1)
    
    run_cmd(cmd=cmd)
    modify_psam(out_prefix, samples)
    if var_pos_offset > 0:
        modify_pvar(out_prefix, var_pos_offset)
    
    return out_prefix

def modify_psam(file_prefix:str, samples:list) -> None:
    df = pd.read_csv(f'{file_prefix}.psam', sep='\t')
    df['#IID'] = samples
    df.to_csv(f'{file_prefix}.psam', sep='\t', index=False)

def modify_pvar(file_prefix:str, var_pos_offset:int) -> None:
    df = pd.read_csv(f'{file_prefix}.pvar', sep='\t')
    df['POS'] = df['POS'] + var_pos_offset
    df.to_csv(f'{file_prefix}.pvar', sep='\t', index=False)

def merge_pgen(pgen_prefix_file:str, out_folder:str, file_prefix:str='') -> None:
    out_prefix = f'{out_folder}/{file_prefix}_merged'
    if os.path.exists(f'{out_prefix}.pgen'):
        print('Already merged')
    
    cmd = f'plink2 --pmerge-list {pgen_prefix_file}'
    cmd += f' --out {out_prefix}'
    cmd += f' --make-pgen'
    
    run_cmd(cmd=cmd)
