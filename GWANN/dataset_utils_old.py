# coding: utf-8
import multiprocessing as mp
import os
import traceback
from functools import partial
from typing import Optional
import csv

import numpy as np
import pandas as pd
import pgenlib as pg
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

from GWANN.utils import vprint


class PGEN2Pandas:
    
    def __init__(self, prefix:str, 
                 sample_subset:Optional[np.ndarray]=None) -> None:
        """Constructor

        Parameters
        ----------
        prefix : str
            Path to pgen+psam+pvar files without the extension.
        sample_subset : Optional[np.ndarray], optional
            Sample subset to limit dosage extraction to only a few
            samples instead of all samples.
        """
        self.psam = pd.read_csv(f'{prefix}.psam', sep='\t')
        if '#FID' in self.psam.columns:
            self.psam.rename(columns={'#FID':'FID'}, inplace=True)
        if 'FID' in self.psam.columns:
            self.psam = self.psam.astype(dtype={'FID':str})
        if '#IID' in self.psam.columns:
            self.psam.rename(columns={'#IID':'IID'}, inplace=True)
        if 'IID' in self.psam.columns:
            self.psam = self.psam.astype(dtype={'IID':str})
        
        if sample_subset is not None:
            self.psam = self.psam.loc[self.psam['IID'].isin(sample_subset)]
            if len(self.psam) != len(sample_subset):
                print(f'Following samples not in pgen file:'+
                    f'{set(sample_subset).difference(set(self.psam["IID"].to_list()))}')
        self.psam.sort_index(inplace=True)
        
        self.pvar = pd.read_csv(f'{prefix}.pvar', sep='\t', 
                                dtype={'#CHROM':str, 'POS':int})
        self.pvar.rename(columns={'#CHROM':'CHROM'}, inplace=True)
        
        self.pgen = pg.PgenReader(
            bytes(f'{prefix}.pgen', 'utf8'), 
            sample_subset=self.psam.index.values.astype(np.uint32))

    def get_samples(self) -> list:
        """Return sample subset.

        Returns
        -------
        list
            List of sample ids. 
        """
        return self.psam['IID'].to_list()

    def get_dosage_matrix(self, chrom:str, start:int, end:int, 
                          SNP_thresh:int=10000) -> Optional[pd.DataFrame]:
        """Extract the dosage values for variants in the interval
        defined by chrom:start-end and return a pandas dataframe of the
        extracted values.

        Parameters
        ----------
        chrom : str
            Chromosome.
        start : int
            Start position on the chromosome.
        end : int
            End position on the chromosome.
        SNP_thresh : int, optional
            Exclude intervals that have number of SNPs > SNP_thresh and
            return None instead of the dataframe, by default 10000

        Returns
        -------
        Optional[pd.DataFrame]
            Pandas dataframe of genotype dosages with vavriants as the
            columns and samples as the rows. If the provided interval
            has > SNP_thresh number of SNPs, return None.
        """

        var_subset = self.pvar.loc[
            (self.pvar['CHROM'] == str(chrom)) &
            (self.pvar['POS'] >= start) & 
            (self.pvar['POS'] <= end)].sort_values('POS')
        if len(var_subset) > SNP_thresh:
            return None
        
        dos = np.empty((len(var_subset), len(self.psam)), dtype=np.float32)
        for i, vari in enumerate(var_subset.index.values):
            self.pgen.read_dosages(vari, dos[i])
        
        dos_df = pd.DataFrame(dos.T.astype(np.float16), 
                              columns=var_subset['ID'].values)
        dos_df.insert(0, 'iid', self.psam['IID'].values)
        
        return dos_df

    @staticmethod
    def remove_indel_multiallelic(dos_df:pd.DataFrame) -> pd.DataFrame:
        """ Remove indels, multiallelic snps or duplicated snps. 

        Parameters
        ----------
        dos_df : pd.DataFrame
            Genotype dosage dataframe

        Returns
        -------
        pd.DataFrame
            Genotype dosage dataframe after filtering variants
        """
        # Remove duplicated SNPs
        print(dos_df.head())
        print(dos_df.T.reset_index().head())
        dos_df = dos_df.T.reset_index().drop_duplicates(keep='first').set_index('index').T
        
        # Remove multiallelic SNPs
        dos_df = dos_df.loc[:, ~dos_df.columns.duplicated(keep=False)]
        # print(dos_df.shape)

        print(dos_df.head())
        dos_df = dos_df[[c for c in dos_df.columns if c.startswith('rs')]]
        
        return dos_df


# Group Train specific functions
def group_ages(ages:np.ndarray, num_grps:int) -> np.ndarray:
    """Function to convert ages into age groups based on frequency of
    each age. The age groups may not be exactly of the same size,
    but the best possible split will be found. 

    Parameters
    ----------
    ages : ndarray
        Array of ages
    num_grps : int
        Number of equally sized (as far as possible) groups to bin the 
        ages into
    
    Returns
    -------
    ndarray
        Array of binned ages. The new age values are integer
        placeholders representing the age group number, not the actual 
        ages
    """
    u_ages = np.sort(np.unique(ages))
    bins = np.append(u_ages, u_ages[-1]+1)
    n, _ = np.histogram(ages, bins=bins)
    cumsum = np.cumsum(n)
    new_ages = ages.copy()
    prev_grp_end = min(u_ages)-1
    
    for g_num in range(num_grps):
        thresh = round((g_num+1)*len(ages)/num_grps)
        grp_end = list(np.array(cumsum >= thresh, dtype=int)).index(1)
    
        diff1 = cumsum[grp_end] - thresh
        diff2 = thresh - cumsum[grp_end - 1]
        grp_end = grp_end if diff1 < diff2 else grp_end-1
        grp_end = u_ages[grp_end]

        condition = (ages <= grp_end) & (ages > prev_grp_end)
        new_ages = np.where(condition, np.repeat(g_num, len(ages)), new_ages)
        
        vprint('Group{}: {}-{} yrs'.format(g_num, prev_grp_end, grp_end))
        prev_grp_end = grp_end
        
    return new_ages

def create_groups(label:str, param_folder:str, phen_cov_path:str, grp_size:int=10,
            oversample:int=10, random_seed:int=82, grp_id_path:str='') -> None:
    """Convert data arrays to grouped data arrays after balancing as
    best as possible for age and sex.

    Parameters
    ----------
    label : str
        Prediction label/phenotype for the dataset.
    param_folder : str
        Path to the folder containing experiment parameters or the
        folder where all additional parameter files should be saved.
    phen_cov_path : str
        Path to the file containing the covariates and phenotype
        information for all individuals in the cohort.    
    grp_size : int, optional
        Size of groups, by default 10
    oversample : int, optional
        Factor to oversample data samples by before forming into
        groups, by default 10
    """
    assert oversample <= grp_size, \
        'Oversample > Group size, cannot ensure that groups have no duplicated iids'
    
    print('\nGrouping ids for for: {}'.format(label))

    if len(grp_id_path) == 0:
        grp_id_path = '{}/group_ids_{}.npz'.format(param_folder, label)
    
    if os.path.isfile(grp_id_path):
        print('Group ids file exists')
        return
    
    df = pd.read_csv(phen_cov_path, sep=' ', comment='#')
    df.drop_duplicates('ID_1', inplace=True)
    df.set_index('ID_1', drop=False, inplace=True)
    
    train_ids_path = '{}/train_ids_{}.csv'.format(param_folder, label)
    train_ids = pd.read_csv(train_ids_path)
    test_ids_path = '{}/test_ids_{}.csv'.format(param_folder, label)
    test_ids = pd.read_csv(test_ids_path)
    
    X = df.loc[train_ids['iid'].values]
    X[label] = train_ids[label].values
    Xt = df.loc[test_ids['iid'].values]
    Xt[label] = test_ids[label].values

    grp_ids = {}
    grp_labels = {}
    for split_name, X_ in {'train':X, 'test':Xt}.items():
        print(f'Split name: {split_name}')
        # split_groups[0] - cont, split_groups[1] - case
        split_groups = [[], []]
        split_group_labels = [[], []]
        
        num_cases = X_.loc[X_[label] == 1].shape[0]
        for j in [0, 1]:
            print(f'Control (0) or Case (1) : {j}')
            iids = X_.loc[X_[label] == j].index.values
            
            ovs_ratio = int(np.round(len(iids)/num_cases))
            ovs = oversample//ovs_ratio
            _, rem = divmod(len(iids)*ovs, grp_size)
            
            print(f'\tNum samples: {len(iids)}')
            print(f'\tGroup size: {grp_size}')
            print(f'\tOversample: {ovs}')
            print(f'\tNum extra samples to be dropped: {rem}')
            
            drop_idxs = []
            if rem != 0:
                drop_iids = np.random.choice(len(iids), size=rem, replace=False)
                if ovs < grp_size:
                    drop_chunks = np.repeat(np.arange(ovs), 1+rem//ovs)[:rem]
                    np.random.seed(random_seed)
                    np.random.shuffle(drop_chunks)
                else:
                    drop_chunks = np.random.choice(ovs, size=rem, replace=False)
                drop_chunks = drop_chunks * len(iids)
                drop_idxs = drop_iids + drop_chunks
                
            chunk_size = grp_size*(len(iids)//grp_size)
            print(f'\tChunk size: {chunk_size}')

            iids = np.tile(iids, ovs)
            iids = np.delete(iids, drop_idxs)
            assert len(iids)%grp_size == 0

            num_chunks = int(np.ceil(len(iids)/chunk_size))
            print(f'\tNum chunks: {num_chunks}')
            print(f'\tLast chunk size: {len(iids)%chunk_size}')

            np.random.seed(random_seed)
            chunk_r_seeds = np.random.randint(0, 10000, size=num_chunks)
        
            for ichunk, rseed in enumerate(chunk_r_seeds):
                
                chunk_iids = iids[ichunk*chunk_size:(ichunk+1)*chunk_size]
                # print(f'\t     Chunk {ichunk} size: {len(chunk_iids)}')
                np.random.seed(rseed)
                np.random.shuffle(chunk_iids)
                
                sex = X_.loc[chunk_iids]['f.31.0.0'].values
                age = X_.loc[chunk_iids]['f.21003.0.0'].values
                vprint('Ages : {}'.format(np.unique(age)))
                age = group_ages(age, num_grps=3)

                n_splits = round(len(chunk_iids)/grp_size)
                # print(f'\t     Num splits: {n_splits}')
                
                # Combine age groups and sex to form stratified groups
                age_sex = np.add(age, sex*3)
                vprint('Age groups: {}'.format(np.unique(age)))
                vprint('Sex: {}'.format(np.unique(sex)))
                vprint('Unique age_sex groups: {}'.format(np.unique(age_sex)))
                
                if n_splits > 1:
                    skf = StratifiedKFold(
                                n_splits=n_splits, 
                                shuffle=True, 
                                random_state=4231)
                    for _, ind in skf.split(chunk_iids, age_sex):
                        assert len(set(chunk_iids[ind])) == grp_size, \
                            f'Group: {sorted(chunk_iids[ind])}, contains duplicate iids'
                        split_groups[j].append(chunk_iids[ind])
                        split_group_labels[j].append(j)
                else:
                    split_groups[j].append(chunk_iids)
                    split_group_labels[j].append(j)
        
        grp_ids[split_name] = np.concatenate((split_groups[0], split_groups[1]))
        grp_labels[split_name] = np.concatenate((split_group_labels[0], split_group_labels[1]))
        print(f'Group ids size: {grp_ids[split_name].shape}')
        print(f'Group labels size: {grp_labels[split_name].shape}')
        n_conts = np.where(grp_labels[split_name] == 0)[0].shape[0]
        n_cases = np.where(grp_labels[split_name] == 1)[0].shape[0]
        assert np.abs(n_cases-n_conts) <= 1

    np.savez(grp_id_path,
        train_grps=grp_ids['train'], train_grp_labels=grp_labels['train'],
        test_grps=grp_ids['test'], test_grp_labels=grp_labels['test'])

def __create_groups(label:str, param_folder:str, phen_cov_path:str, grp_size:int=10,
            oversample:int=10, random_seed:int=82, grp_id_path:str='') -> None:
    """Convert data arrays to grouped data arrays after balancing as
    best as possible for age and sex.

    Parameters
    ----------
    label : str
        Prediction label/phenotype for the dataset.
    param_folder : str
        Path to the folder containing experiment parameters or the
        folder where all additional parameter files should be saved.
    phen_cov_path : str
        Path to the file containing the covariates and phenotype
        information for all individuals in the cohort.    
    grp_size : int, optional
        Size of groups, by default 10
    oversample : int, optional
        Factor to oversample data samples by before forming into
        groups, by default 10
    """
    assert oversample <= grp_size, \
        'Oversample > Group size, cannot ensure that groups have no duplicated iids'
    
    print('\nGrouping ids for for: {}'.format(label))

    if len(grp_id_path) == 0:
        grp_id_path = '{}/group_ids_{}.npz'.format(param_folder, label)
    
    if os.path.isfile(grp_id_path):
        print('Group ids file exists')
        return
    
    df = pd.read_csv(phen_cov_path, sep=' ', comment='#')
    df.set_index('ID_1', drop=False, inplace=True)
    
    train_ids_path = '{}/train_ids_{}.csv'.format(param_folder, label)
    train_ids = pd.read_csv(train_ids_path)
    test_ids_path = '{}/test_ids_{}.csv'.format(param_folder, label)
    test_ids = pd.read_csv(test_ids_path)
    
    X = df.loc[train_ids['iid'].values]
    X[label] = train_ids[label].values
    Xt = df.loc[test_ids['iid'].values]
    Xt[label] = test_ids[label].values

    grp_ids = {}
    grp_labels = {}
    for split_name, X_ in {'train':X, 'test':Xt}.items():
        print(f'Split name: {split_name}')
        # split_groups[0] - cont, split_groups[1] - case
        split_groups = [[], []]
        split_group_labels = [[], []]
        for j in [0, 1]:
            print(f'Control (0) or Case (1) : {j}')
            iids = X_.loc[X_[label] == j].index.values
            rem = (len(iids)*oversample)%grp_size
            
            print(f'\tNum samples: {len(iids)}')
            print(f'\tGroup size: {grp_size}')
            print(f'\tNum extra samples to be dropped: {rem}')
            
            drop_idxs = []
            if rem != 0:
                drop_iids = np.random.choice(len(iids), size=rem, replace=False)
                if oversample < grp_size:
                    drop_chunks = np.repeat(np.arange(oversample), 1+rem//oversample)[:rem]
                    np.random.seed(random_seed)
                    np.random.shuffle(drop_chunks)
                else:
                    drop_chunks = np.random.choice(oversample, size=rem, replace=False)
                drop_chunks = drop_chunks * len(iids)
                drop_idxs = drop_iids + drop_chunks
                
            chunk_size = grp_size*(len(iids)//grp_size)
            print(f'\tChunk size: {chunk_size}')

            t = len(iids) 
            iids = np.tile(iids, oversample)
            iids = np.delete(iids, drop_idxs)
            assert len(iids)%grp_size == 0

            num_chunks = int(np.ceil(len(iids)/chunk_size))
            print(f'\tNum chunks: {num_chunks}')
            print(f'\tLast chunk size: {len(iids)%chunk_size}')

            np.random.seed(random_seed)
            chunk_r_seeds = np.random.randint(0, 10000, size=num_chunks)
        
            for ichunk, rseed in enumerate(chunk_r_seeds):
                
                chunk_iids = iids[ichunk*chunk_size:(ichunk+1)*chunk_size]
                print(f'\t     Chunk {ichunk} size: {len(chunk_iids)}')
                np.random.seed(rseed)
                np.random.shuffle(chunk_iids)
                
                sex = X_.loc[chunk_iids]['f.31.0.0'].values
                age = X_.loc[chunk_iids]['f.21003.0.0'].values
                vprint('Ages : {}'.format(np.unique(age)))
                age = group_ages(age, num_grps=3)

                n_splits = round(len(chunk_iids)/grp_size)
                print(f'\t     Num splits: {n_splits}')
                
                # Combine age groups and sex to form stratified groups
                age_sex = np.add(age, sex*3)
                vprint('Age groups: {}'.format(np.unique(age)))
                vprint('Sex: {}'.format(np.unique(sex)))
                vprint('Unique age_sex groups: {}'.format(np.unique(age_sex)))
                
                if n_splits > 1:
                    skf = StratifiedKFold(
                                n_splits=n_splits, 
                                shuffle=True, 
                                random_state=4231)
                    for _, ind in skf.split(chunk_iids, age_sex):
                        assert len(set(chunk_iids[ind])) == grp_size, \
                            f'Group: {sorted(chunk_iids[ind])}, contains duplicate iids'
                        split_groups[j].append(chunk_iids[ind])
                        split_group_labels[j].append(j)
                else:
                    split_groups[j].append(chunk_iids)
                    split_group_labels[j].append(j)
        
        grp_ids[split_name] = np.concatenate((split_groups[0], split_groups[1]))
        grp_labels[split_name] = np.concatenate((split_group_labels[0], split_group_labels[1]))
        print(f'\tGroup ids size: {grp_ids[split_name].shape}')
        print(f'\tGroup labels size: {grp_labels[split_name].shape}')

    np.savez(grp_id_path,
        train_grps=grp_ids['train'], train_grp_labels=grp_labels['train'],
        test_grps=grp_ids['test'], test_grp_labels=grp_labels['test'])

def _create_groups(label:str, param_folder:str, phen_cov_path:str, grp_size:int=10,
            train_oversample:int=10, test_oversample:int=10, random_seed:int=82, 
            grp_id_path:str='') -> None:
    """Convert data arrays to grouped data arrays after balancing as
    best as possible for age and sex.

    Parameters
    ----------
    label : str
        Prediction label/phenotype for the dataset.
    param_folder : str
        Path to the folder containing experiment parameters or the
        folder where all additional parameter files should be saved.
    phen_cov_path : str
        Path to the file containing the covariates and phenotype
        information for all individuals in the cohort.    
    grp_size : int, optional
        Size of groups, by default 10
    train_oversample : int, optional
        Factor to oversample all train data samples by before forming into
        groups, by default 10
    test_oversample : int, optional
        Factor to oversample all test data samples by before forming into
        groups, by default 10

    """
    print('\nGrouping ids for for: {}'.format(label))

    if len(grp_id_path) == 0:
        grp_id_path = '{}/group_ids_{}.npz'.format(param_folder, label)
    
    if os.path.isfile(grp_id_path):
        print('Group ids file exists')
        return
    
    df = pd.read_csv(phen_cov_path, sep=' ', comment='#')
    df.set_index('ID_1', drop=False, inplace=True)
    
    train_ids_path = '{}/train_ids_{}.csv'.format(param_folder, label)
    train_ids = pd.read_csv(train_ids_path)['iid'].values
    test_ids_path = '{}/test_ids_{}.csv'.format(param_folder, label)
    test_ids = pd.read_csv(test_ids_path)['iid'].values
    
    X = df.loc[train_ids]
    y = df.loc[train_ids][label].values
    Xt = df.loc[test_ids]
    yt = df.loc[test_ids][label].values

    grp_ids = []
    grp_labels = []
    for X_, y_, over in [(X, y, train_oversample), (Xt, yt, test_oversample)]:
        
        case = np.where(y_ == 1)[0]
        cont = np.where(y_ == 0)[0]
        
        # Randomly oversample and interleave the individuals
        
        case = np.repeat(case, over)
        np.random.seed(random_seed)
        np.random.shuffle(case)
        cont = np.repeat(cont, over)
        np.random.seed(random_seed)
        np.random.shuffle(cont)

        # Remove extra samples that will not form a group of the
        # expected size
        case_num = len(case) - len(case)%grp_size
        cont_num = len(cont) - len(cont)%grp_size
        np.random.seed(random_seed)
        case = np.random.choice(case, case_num, replace=False)
        np.random.seed(random_seed)
        cont = np.random.choice(cont, cont_num, replace=False)

        # Create groups balanced on age and sex (as close as possible)
        # Xg[0] - Case, Xg[1] - Control
        Xg = [[], []]
        sex = X_['f.31.0.0'].values
        age = X_['f.21003.0.0'].values
        vprint('Ages : {}'.format(np.unique(age)))
        age = group_ages(age, num_grps=3)
        for j, idxs in enumerate([case, cont]):
            n_splits = round(len(idxs)/grp_size)
            X_idxs = X_.iloc[idxs, :]
            sex_idxs = sex[idxs]
            age_idxs = age[idxs]

            # Combine age groups and sex to form stratified groups
            age_sex = np.add(age_idxs, sex_idxs*3)
            vprint('Age groups: {}'.format(np.unique(age_idxs)))
            vprint('Sex: {}'.format(np.unique(sex_idxs)))
            vprint('Unique age_sex groups: {}'.format(np.unique(age_sex)))
            
            skf = StratifiedKFold(
                        n_splits=n_splits, 
                        shuffle=True, 
                        random_state=4231)
            for _, ind in skf.split(np.zeros(len(idxs)), age_sex):
                Xg[j].append(X_idxs.iloc[ind].index.values)
                
        grp_ids.append(np.concatenate((Xg[0], Xg[1])))
        grp_labels.append(np.concatenate((np.ones(len(Xg[0])), np.zeros(len(Xg[1])))))

    np.savez(grp_id_path,
        train_grps=grp_ids[0],
        train_grp_labels=grp_labels[0],
        test_grps=grp_ids[1],
        test_grp_labels=grp_labels[1])

# Data creation and loading functions
def load_data(pg2pd:Optional[PGEN2Pandas], phen_cov:Optional[pd.DataFrame], 
              gene:str, chrom:str, start:int, end:int, buffer:int, label:str, 
              sys_params:dict, covs:list, SNP_thresh:int=10000, 
              only_covs:bool=False, preprocess:bool=True,
              lock:Optional[mp.Lock]=None) -> Optional[tuple]:
    """Load data, balance it and obtain train-test splits before
    training. If the preprocess argument is False (by default True), 
    this function will simply create and save the data to disk and 
    return None.

    Parameters
    ----------
    pg2pd : PGEN2Pandas, optional
        An object of the type PGEN2Pandas which can be passed a
        chromosome, a start position, and an end position to extract
        the SNP dosages from a PGEN (PLINK 2.0) file for all samples.
    phen_cov : pd.DataFrame, optional
        Dataframe containing covariate and phenotype information
    gene : list
        List of genes to include in the dataset. For a single gene 
        pass as a list of a single str
    chrom : list
        List of chromosomes of the genes to include in the dataset. 
        For a single gene pass as a list of a single str.
    start: int
        Start position of the gene. This should not include the buffer
        because the final start position will be start-buffer.
    end: int
        End position of the gene. This should not include the buffer
        because the final end position will be end+buffer.
    buffer: int
        Base-pair buffer to consider on each side of the gene.
    label : str
        Prediction label/phenotype for the dataset.
    sys_params : dict
        Dictionary containing system specific parameters - necessary
        foder paths.
    covs : list
       List of covariates
    SNP_thresh : int, optional
        Maximum number of SNPs. Genes with SNPs greater 
        than this will be dropped, by default 10000
    only_covs : bool, optional
        Return a dataframe of only covariates, without any SNPs.
    preprocess: bool, optional
        Whether to preprocess data or notm by default True. If the
        objective is simply to create and save data to dislk, pass False.
    lock : mp.Lock, optional
        Lock object to prevent issues with concurrent access during read
        and write. Pass None to prevent logging of data stats, by default None.

    Returns
    -------
    tuple
        Data tuple containing the following in the respective
        indices:
        0 - Balanced training data (ndarray)
        1 - Balanced training labels (ndarray)
        2 - Balanced testing data (ndarray)
        3-  Balanced testing label (ndarray)
        4 - Array containing class weights based on training labels
        (ndarray)
        5 - Names of each column in the data arrays (list)
        6 - Number of SNPs in the data arrays (int)
    """
    log_creation = lock is not None
    save_data = True

    test_ids_f = f'{sys_params["PARAMS_PATH"]}/test_ids_{label}.csv'
    test_ids_df = pd.read_csv(test_ids_f, dtype={'iid':str})
    test_labels = test_ids_df[label].to_list()
    test_ids = test_ids_df['iid'].to_list()
    
    train_ids_f = f'{sys_params["PARAMS_PATH"]}/train_ids_{label}.csv'
    train_ids_df = pd.read_csv(train_ids_f, dtype={'iid':str})
    train_labels = train_ids_df[label].to_list()
    train_ids = train_ids_df['iid'].to_list()

    data_path = (f'{sys_params["DATA_BASE_FOLDER"]}/'+
                f'chr{chrom}_{gene}_{buffer}bp_{label}.csv')

    if os.path.exists(data_path):
        data_mat = pd.read_csv(data_path, index_col=0, comment='#')
        data_mat.index = data_mat.index.astype(str)
        data_mat = data_mat.loc[train_ids+test_ids]
        save_data = False
    else:
        data_mat = pg2pd.get_dosage_matrix(chrom=chrom, start=start-buffer, 
                                        end=end+buffer, SNP_thresh=SNP_thresh)
        if data_mat is None:
            return None
        
        data_mat.set_index('iid', inplace=True)
        data_mat = data_mat.loc[train_ids+test_ids]
        data_mat = pd.merge(data_mat, phen_cov[covs].loc[train_ids+test_ids], 
                            left_index=True, right_index=True)
    
    data_mat.loc[train_ids+test_ids, label] = train_labels+test_labels
    
    data_mat = data_mat.loc[:,~data_mat.columns.duplicated()]

    assert not np.any(data_mat.columns.duplicated()), \
        f'Data has duplicated columns: {data_mat.columns[data_mat.columns.duplicated()]}'

    assert not np.any(pd.isna(data_mat)), \
            f'[{gene}]: Dataframe contains NaN values'
    
    if only_covs:
        data_mat = data_mat[covs+[label,]]

    # Save dataframe to disk
    if save_data:
        data_mat.to_csv(data_path)
        print(f'[{gene}] Data written.')

    train_df = data_mat.loc[train_ids]
    test_df = data_mat.loc[test_ids]

    data_tuple = None
    if preprocess:
        data_tuple = preprocess_data(train_df=train_df, test_df=test_df, 
                                    label=label, covs=covs, sys_params=sys_params)    
        num_snps = data_tuple[-1]
    else:
        data_cols = [c for c in train_df.columns if c!=label]
        snp_cols = [c for c in data_cols if c not in covs]
        num_snps = len(snp_cols)

    if log_creation and save_data:
        try:
            lock.acquire()
            data_stats_f = f'{sys_params["RUNS_BASE_FOLDER"]}/dataset_stats.csv'
            header = ['Gene', 'chrom', 'start', 'end', 'num_snps', 'label']
            data_info = [gene, chrom, str(start-buffer), str(end+buffer), num_snps, label]
            if os.path.isfile(data_stats_f):
                rows = [data_info]
            else:
                rows = [header, data_info]
        
            with open(data_stats_f, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
        
        except Exception as e:
            raise e
        finally:
            lock.release()
    
    return data_tuple

def load_win_data(gene:str, win:int, chrom:str, buffer:int, label:str, 
                  sys_params:dict, covs:list, 
                  only_covs:bool=False) -> Optional[tuple]:
    """Load data from disk. This function is a quick data loading if the
    data for each gene has already been split into windows.

    Parameters
    ----------
    gene : list
        List of genes to include in the dataset. For a single gene 
        pass as a list of a single str
    chrom : list
        List of chromosomes of the genes to include in the dataset. 
        For a single gene pass as a list of a single str.
    buffer: int
        Base-pair buffer to consider on each side of the gene.
    label : str
        Prediction label/phenotype for the dataset.
    sys_params : dict
        Dictionary containing system specific parameters - necessary
        foder paths.
    covs : list
       List of covariates
    only_covs : bool, optional
        Return a dataframe of only covariates, without any SNPs.
    
    Returns
    -------
    tuple
        Data tuple containing the following in the respective
        indices:
        0 - Balanced training data (ndarray)
        1 - Balanced training labels (ndarray)
        2 - Balanced testing data (ndarray)
        3-  Balanced testing label (ndarray)
        4 - Array containing class weights based on training labels
        (ndarray)
        5 - Names of each column in the data arrays (list)
        6 - Number of SNPs in the data arrays (int)
    """
    test_ids_f = f'{sys_params["PARAMS_PATH"]}/test_ids_{label}.csv'
    test_ids_df = pd.read_csv(test_ids_f, dtype={'iid':str})
    test_labels = test_ids_df[label].to_list()
    test_ids = test_ids_df['iid'].to_list()
    
    train_ids_f = f'{sys_params["PARAMS_PATH"]}/train_ids_{label}.csv'
    train_ids_df = pd.read_csv(train_ids_f, dtype={'iid':str})
    train_labels = train_ids_df[label].to_list()
    train_ids = train_ids_df['iid'].to_list()

    data_path = (f'{sys_params["DATA_BASE_FOLDER"]}/wins/' +
                f'chr{chrom}_{gene}_{win}_{buffer}bp_{label}.csv')

    data_mat = pd.read_csv(data_path, index_col=0, comment='#')
    data_mat.index = data_mat.index.astype(str)
    data_mat.loc[train_ids+test_ids, label] = train_labels+test_labels
    data_mat = data_mat.loc[:,~data_mat.columns.duplicated()]

    assert not np.any(data_mat.columns.duplicated()), \
        f'Data has duplicated columns: {data_mat.columns[data_mat.columns.duplicated()]}'

    train_df = data_mat.loc[train_ids]
    test_df = data_mat.loc[test_ids]
    
    if set(covs).difference(set(data_mat.columns)):
        print(f'[{gene}]: Covariates in data do not match cov yaml file')
        return

    if only_covs:
        train_df = train_df[covs+[label,]]
        test_df = test_df[covs+[label,]]
    
    assert not np.any(pd.isna(train_df)), \
            f'[{gene}]: Train dataframe contains NaN values'
    assert not np.any(pd.isna(test_df)), \
            f'[{gene}]: Test dataframe contains NaN values'

    data_tuple = preprocess_data(train_df=train_df, test_df=test_df, label=label,
                                 covs=covs, sys_params=sys_params)

    return data_tuple

def preprocess_data(train_df:pd.DataFrame, test_df:pd.DataFrame, label:str, 

                    covs:list, sys_params:dict) -> tuple:
    """Given a training and testing pandas dataframe, this function
    scales features between 0 and 1, and converts samples into 'grouped
    samples' for group training. 

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split dataframe.
    test_df : pd.DataFrame
        Testing split dataframe.
    label : str
        Prediction label/phenotype for the dataset.
    covs : list
        List of covariates.
    sys_params : dict
        Dictionary containing system specific parameters - necessary
        foder paths.

    Returns
    -------
    tuple
        Data tuple containing the following in the respective
        indices:
        0 - Balanced training data (ndarray)
        1 - Balanced training labels (ndarray)
        2 - Balanced testing data (ndarray)
        3-  Balanced testing label (ndarray)
        4 - Array containing class weights based on training labels
        (ndarray)
        5 - Names of each column in the data arrays (list)
        6 - Number of SNPs in the data arrays (int)
    """
    data_cols = [c for c in train_df.columns if c!=label]
    snp_cols = [c for c in data_cols if c not in covs]
    num_snps = len(snp_cols)
    
    # Scale each feature between 0 and 1
    mm_scaler = MinMaxScaler(feature_range=(0, 1))
    mm_scaler.fit(train_df)
    scaled_train_df = mm_scaler.transform(train_df)
    train_df.iloc[:, num_snps:-1] = scaled_train_df[:, num_snps:-1]
    
    scaled_test_df = mm_scaler.transform(test_df)
    test_df.iloc[:, num_snps:-1] = scaled_test_df[:, num_snps:-1]
    
    # Scale SNPs between -1 and 1 to match scalling of GWANNv1
    # mm_scaler = MinMaxScaler(feature_range=(-1, 1))
    # mm_scaler.fit(train_df)
    # scaled_train_df = mm_scaler.transform(train_df)
    # train_df.iloc[:, :num_snps] = scaled_train_df[:, :num_snps]
    
    # scaled_test_df = mm_scaler.transform(test_df)
    # test_df.iloc[:, :num_snps] = scaled_test_df[:, :num_snps]

    # Convert data into grouped samples
    # grps = np.load(f'{sys_params["GROUP_IDS_PATH"]}')
    # grp_size = grps['train_grps'].shape[1]
    # train_grps = np.asarray(grps['train_grps'].flatten(), dtype=int).astype(str)
    # test_grps = np.asarray(grps['test_grps'].flatten(), dtype=int).astype(str)
    
    # X = train_df.loc[train_grps][data_cols].to_numpy()
    # X = np.reshape(X, (-1, grp_size, X.shape[-1]))
    # y = grps['train_grp_labels']

    # X_test = test_df.loc[test_grps][data_cols].to_numpy()
    # X_test = np.reshape(X_test, (-1, grp_size, X_test.shape[-1]))
    # y_test = grps['test_grp_labels']
    
    X = train_df[data_cols].to_numpy()
    y = train_df[label].values
    
    X_test = test_df[data_cols].to_numpy()
    y_test = test_df[label].values
    
    class_weights = compute_class_weight(class_weight='balanced', 
                                        classes=[0, 1], y=y)

    return X, y, X_test, y_test, class_weights, data_cols, num_snps

def balance_by_agesex(data:pd.DataFrame, label_col:str, 
                      control_prop:float=1.0) -> pd.DataFrame:
    """Function to return a 1:1 balanced version of a dataset. Balancing is 
    done to ensure the same resultant distribution for age and sex in 
    case and control.

    Parameters
    ----------
    data : Pandas.DataFrame
        CSV data file
    label_col : str
        Column in data that corresponds to the label to balance for
    control_prop : float
        Proportion of controls wrt cases, by default 1.0

    Returns 
    -------
    Pandas.DataFrame
        Balanced data of the same structure

    """

    # f.21003.0.0 - Age, f.31.0.0 - Sex 
    x, y = np.unique(data[['f.21003.0.0', 'f.31.0.0']], axis=0, 
        return_counts=True)
    case_age = {'{},{}'.format(a[0],a[1]):0 for a in x}

    # Find the frequency of different ages in cases
    x, y = np.unique(data.loc[data[label_col]==1][['f.21003.0.0', 'f.31.0.0']], 
                        axis=0, return_counts=True)
    for xi, yi in zip(x,y):
        case_age['{},{}'.format(xi[0],xi[1])] = yi

    # Randomly sample controls from same distribution of age and sex as in case
    cases = data.loc[data[label_col]==1]
    conts = pd.DataFrame()
    cont_age = {'{},{}'.format(a[0],a[1]):0 for a in x}
    for ca in case_age.keys():
        a, s = float(ca.split(',')[0]), float(ca.split(',')[1])
        tmp_df = data.loc[(data[label_col] == 0) & 
                            (data['f.21003.0.0'] == a) & 
                            (data['f.31.0.0'] == s)]
        if len(tmp_df) == 0:
            continue
        required_grp_cnt = int(case_age[ca]*control_prop)
        tmp_df = tmp_df.sample(min(tmp_df.shape[0], required_grp_cnt), 
                               replace=False, random_state=7639)
        conts = conts.append(tmp_df)
        del(tmp_df)

    # If the count of controls is not the required count after trying to
    # balance for age and sex, randomly sample the extra required controls
    required_cnt = int(np.sum(list(case_age.values()))*control_prop)
    if len(conts) < required_cnt:
        remaining_conts = data.loc[
            (data[label_col] == 0) & 
            (~data.index.isin(conts.index))]
        sample_cnt = min(remaining_conts.shape[0], required_cnt-len(conts))
        conts = pd.concat((
            conts, 
            remaining_conts.sample(sample_cnt, replace=False, random_state=7639)))

    balanced_data = cases.append(conts)
    vprint('Case count: {}'.format
            (balanced_data[balanced_data[label_col] == 1].shape[0]))
    vprint('Control count: {}'.format
            (balanced_data[balanced_data[label_col] == 0].shape[0]))
    
    return balanced_data

# Functions to create and split data of a chromosome in parallel.
# Useful to create and save all data before running NN training
# pipeline. 
def write_and_return_data(gene_dict:dict, chrom:str, lock:Optional[mp.Lock], 
                sys_params:dict, covs:list, buffer:int, label:str, 
                only_covs:bool=False, SNP_thresh:int=10000, 
                preprocess:bool=False, ret_data:bool=False) -> Optional[tuple]:
    """Invokes the data creation pipeline for a set of genes.

    Parameters
    ----------
    gene_dict : dict
        Dictionaty containing gene name, chromosome, start position and 
        end position. Each value should be a list. 
        Eg. {'names':['G1', 'G2'], 'chrom':['19', '19'], 'start':[100,200],
        'end':[150, 250]}
    chrom : str
        Chromosome name.
    lock : multiprocessing.Lock
        Lock object to aid process synchronisation. 
    sys_params : dict
        Dictionary of system parameters eg. path to data, path to test
        ids etc.
    covs : list
        List of covariates.
    buffer : int
        Number of flanking base pairs to consider as part of the gene 
        while creating the data, by default 2500.
    label : str
        Phenotype label.
    only_covs : bool, optional
        Whether to create data using only covariates (True) or covariates and
        SNPs (False), by default False.
    SNP_thresh : int, optional
        Maximum number of SNPs to consider. If a gene has more than this
        many SNPs, the file will not be created for the gene, by defailt
        10000.
    preprocess: bool, optional
        Whether to preprocess data or notm by default True. If the
        objective is simply to create and save data to dislk, pass False.
    ret_data : bool, optional
        Whether to return the created data or not, by default False.
    
    Returns
    -------
    tuple or int
        if ret_data == True, tuple of data files 
            (X, y, X_test, y_test, class_weights, data_cols, num_snps)
        if ret_data == False, 0
    """
   
    pgen_prefix = f'{sys_params["RAW_BASE_FOLDER"][chrom]}/UKB_chr{chrom}'
    test_ids_f = f'{sys_params["PARAMS_PATH"]}/test_ids_{label}.csv'
    test_ids_df = pd.read_csv(test_ids_f, dtype={'iid':str})
    test_ids = test_ids_df['iid'].to_list()
    
    train_ids_f = f'{sys_params["PARAMS_PATH"]}/train_ids_{label}.csv'
    train_ids_df = pd.read_csv(train_ids_f, dtype={'iid':str})
    train_ids = train_ids_df['iid'].to_list()
    
    pg2pd = PGEN2Pandas(pgen_prefix, sample_subset=train_ids+test_ids)
    
    phen_cov = pd.read_csv(sys_params['PHEN_COV_PATH'], 
                           sep=' ', dtype={'ID_1':str}, comment='#')
    phen_cov = phen_cov.rename(columns={'ID_1':'iid'})
    phen_cov.index = phen_cov['iid']
    
    for i, gene in enumerate(gene_dict['names']):
        data = None
        try:
            data = load_data(pg2pd=pg2pd, phen_cov=phen_cov, gene=gene, 
                        chrom=chrom, start=gene_dict['start'][i], 
                        end=gene_dict['end'][i], buffer=buffer, label=label, 
                        sys_params=sys_params, covs=covs, SNP_thresh=SNP_thresh, 
                        only_covs=only_covs, preprocess=preprocess, lock=lock)
        except Exception:
            print(f'[{gene}] - Data creating error. Check {sys_params["DATA_LOGS"]}')
            if lock is not None:
                lock.acquire()
                with open(sys_params['DATA_LOGS'], 'a') as f:
                    f.write(f'\n{gene}\n')
                    f.write('--------------------\n')
                    f.write(traceback.format_exc())
                lock.release()

        if data is None:
            continue
        print('Train data shape for {}: {}'.format(gene, data[0].shape))
        
    if ret_data:
        return data
    else:
        return None
    
def create_data_for_run(label:str, chrom:str, glist:Optional[list], 
                        sys_params:dict, covs:list, gene_map_file:str, 
                        buffer:int=2500, SNP_thresh:int=10000, 
                        preprocess:bool=False, num_procs_per_chrom:int=2) -> None:
    """Create data files for a set of genes on a chromosome.

    Parameters
    ----------
    label : str
        Phenotyoe label.
    chrom : str, int
        Chromosome.
    glist : Optional[list]
        List of gene symbols to create data for. To create data for all
        genes on the chromosome, pass None.
    sys_params : dict
        Dictionary of system parameters eg. path to data, path to test
        ids etc.
    covs : list
        List of covariates.
    gene_map_file : str
        Path to the file containing the map of genes to their annotations.
    buffer : int, optional
        Number of flanking base pairs to consider as part of the gene 
        while creating the data, by default 2500.
    SNP_thresh: int, optional
        Maximum number of SNPs allowed in data. If number of SNPs
        exceeds this value, data will not be created, by default 10000.
    preprocess: bool, optional
        Whether to preprocess data or notm by default True. If the
        objective is simply to create and save data to dislk, pass False.
    num_procs_per_chrom : int, optional
        Number of CPU cores to assign for the task, by default 2.
    """

    genes_df = pd.read_csv(gene_map_file)
    genes_df.drop_duplicates(['symbol'], inplace=True)
    genes_df = genes_df.astype({'chrom':str})
    genes_df.set_index('symbol', drop=False, inplace=True)
    genes_df = genes_df.loc[genes_df['chrom'] == str(chrom)]
    if len(glist) is not None:
        genes_df = genes_df.loc[genes_df['symbol'].isin(glist)]
    
    if len(genes_df) == 0:
        print('No genes found with given chrom and glist')
        return
    
    gdict = {'names':[], 'start':[], 'end':[]}
    for _, r in genes_df.iterrows():
        g = r['symbol']
        s = r['start']
        e = r['end']
        data_path = (f'{sys_params["DATA_BASE_FOLDER"]}/'+
                    f'chr{chrom}_{g}_{buffer}bp_{label}.csv')
        if not os.path.isfile(data_path):
            gdict['names'].append(g)
            gdict['start'].append(s)
            gdict['end'].append(e)
        else:
            print(f'Data file for {g} exists at {data_path}')
            continue
        
    if len(gdict['names']) == 0:
        print('No genes left to create data for.')
        return
    
    print(f'Creating data for {len(gdict["names"])} genes')
    
    ds = []
    num_procs = min(len(gdict['names']), num_procs_per_chrom)
    split_idxs = np.array_split(np.arange(len(gdict['names'])), num_procs)
    for sidxs in split_idxs:
        d_win = {}
        for k in gdict.keys():
            d_win[k] = [gdict[k][si] for si in sidxs]
        ds.append(d_win)

    lock = mp.Manager().Lock()
    with mp.Pool(num_procs) as pool:
        par_func = partial(write_and_return_data, 
            chrom=chrom, lock=lock, sys_params=sys_params, covs=covs, 
            buffer=buffer, label=label, ret_data=False, SNP_thresh=SNP_thresh, 
            preprocess=preprocess)
        pool.map(par_func, ds)
        pool.close()
        pool.join()

def split(genes:list, covs:list, label:str, read_base:str, 
          write_base:str) -> None:
    """Split gene datasets into windows of 50 SNPs.

    Parameters
    ----------
    genes : list
        List of gene data file names.
    covs : list
        List of covariate columns in the data.
    label : str
        Name of the label/phenotype in the data.
    read_base : str
        Base folder to read data from. 
    write_base : str
        Base folder to write data to. 
    """
    for gene_file in genes:
        df_path = f'{read_base}/{gene_file}'
        df = pd.read_csv(df_path, index_col=0, comment='#')
        data_cols = [c for c in df.columns.to_list() if c != label]
        snp_cols = [c for c in data_cols if c not in covs]
        snp_win = 50
        split_snp_cols = np.array_split(snp_cols, np.arange(snp_win, len(snp_cols), snp_win))
        
        for win, split_snps in enumerate(split_snp_cols):
            split_f = gene_file.split('_')
            split_f[1] = f'{split_f[1]}_{win}'
            f_win = f'{write_base}/{"_".join(split_f)}'
            win_cols = list(split_snps) + covs + [label,]
            df[win_cols].to_csv(f_win)
