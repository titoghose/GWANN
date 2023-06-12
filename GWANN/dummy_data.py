import subprocess

def run_cmd(cmd:str) -> None:
    subprocess.run(cmd, shell=True)

def dummy_plink(num_samples:int, num_snps:int, dosage_freq:float, 
                out_folder:str) -> None:
    out_prefix = f'{out_folder}/dummy_{num_snps}snps_{dosage_freq:.4f}dosfreq'
    cmd = f'plink2 --dummy {num_samples} {num_snps} acgt'
    cmd += f' dosage-freq={dosage_freq}'
    cmd += f' --out {out_prefix}'
    cmd += f' --make-pgen'
    run_cmd(cmd=cmd)
    return out_prefix