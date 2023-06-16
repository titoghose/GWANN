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
        
        print('Group{}: {}-{} yrs'.format(g_num, prev_grp_end, grp_end))
        prev_grp_end = grp_end
        
    return new_ages

def create_groups(label:str, param_folder:str, phen_cov_path:str, grp_size:int=10,
            train_oversample:int=10, test_oversample:int=10) -> None:
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
    
    group_id_path = '{}/group_ids_{}.npz'.format(param_folder, label)
    if os.path.isfile(group_id_path):
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
        np.random.seed(82)
        # 3 np.random.seed(192)
        # 2 np.random.seed(8376)
        # 1 np.random.seed(1763)
        np.random.shuffle(case)
        cont = np.repeat(cont, over)
        np.random.seed(82)
        # 3 np.random.seed(192)
        # 2 np.random.seed(8376)
        # 1 np.random.seed(1763)
        np.random.shuffle(cont)

        # Remove extra samples that will not form a group of the
        # expected size
        case_num = len(case) - len(case)%grp_size
        cont_num = len(cont) - len(cont)%grp_size
        np.random.seed(102)
        # 3 np.random.seed(1108)
        # 2 np.random.seed(1763)
        # 1 np.random.seed(8983)
        case = np.random.choice(case, case_num, replace=False)
        np.random.seed(102)
        # 3 np.random.seed(1108)
        # 2 np.random.seed(1763)
        # 1 np.random.seed(8983)
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

    np.savez(group_id_path,
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

    test_ids_f = sys_params['TEST_IDS_PATH']
    test_ids = pd.read_csv(test_ids_f, dtype={'iid':str})['iid'].to_list()
    
    train_ids_f = sys_params['TRAIN_IDS_PATH']
    train_ids = pd.read_csv(train_ids_f, dtype={'iid':str})['iid'].to_list()

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
        data_mat = pd.merge(data_mat, phen_cov[covs+[label,]].loc[train_ids+test_ids], 
                            left_index=True, right_index=True)
    
    edu_col = 'f.6138'
    if edu_col in covs:
        # print(data_mat.loc[data_mat[edu_col].isna()])
        data_mat.loc[data_mat[edu_col] == -7, edu_col] = np.nan
        data_mat.loc[data_mat[edu_col] == -3, edu_col] = np.nan
        data_mat.loc[data_mat[edu_col].isna(), edu_col+'_missing'] = 1
        data_mat.loc[~data_mat[edu_col].isna(), edu_col+'_missing'] = 0
    # print(data_mat.loc[data_mat[edu_col].isna()])
    
    assert not np.any(data_mat.columns.duplicated()), \
        f'Data has duplicated columns: {data_mat.columns[data_mat.columns.duplicated()]}'

    assert not np.any(pd.isna(data_mat[[c for c in data_mat.columns if c!=edu_col]])), \
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
        num_snps = train_df.shape[-1] - len(covs) - 1

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
    test_ids_f = sys_params['TEST_IDS_PATH']
    test_ids = pd.read_csv(test_ids_f, dtype={'iid':str})['iid'].to_list()
    
    train_ids_f = sys_params['TRAIN_IDS_PATH']
    train_ids = pd.read_csv(train_ids_f, dtype={'iid':str})['iid'].to_list()

    data_path = (f'{sys_params["DATA_BASE_FOLDER"]}/wins/' +
                f'chr{chrom}_{gene}_{win}_{buffer}bp_{label}.csv')

    data_mat = pd.read_csv(data_path, index_col=0, comment='#')
    data_mat.index = data_mat.index.astype(str)
    train_df = data_mat.loc[train_ids]
    test_df = data_mat.loc[test_ids]
        
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
    num_snps = len(data_cols) - len(covs)
    
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
    
    # Fill missing values for f.6138
    train_df.fillna(-1, inplace=True)
    test_df.fillna(-1, inplace=True)

    # Convert data into grouped samples
    grps = np.load(f'{sys_params["PARAMS_PATH"]}/group_ids_{label}.npz')
    grp_size = grps['train_grps'].shape[1]
    train_grps = np.asarray(grps['train_grps'].flatten(), dtype=int).astype(str)
    test_grps = np.asarray(grps['test_grps'].flatten(), dtype=int).astype(str)
    
    X = train_df.loc[train_grps][data_cols].to_numpy()
    X = np.reshape(X, (-1, grp_size, X.shape[-1]))
    y = grps['train_grp_labels']

    X_test = test_df.loc[test_grps][data_cols].to_numpy()
    X_test = np.reshape(X_test, (-1, grp_size, X_test.shape[-1]))
    y_test = grps['test_grp_labels']
    
    class_weights = compute_class_weight(class_weight='balanced', 
                                        classes=np.unique(y), y=y)

    return X, y, X_test, y_test, class_weights, data_cols, num_snps

def balance_by_agesex(data:pd.DataFrame, label_col:str) -> pd.DataFrame:
    """Function to return a 1:1 balanced version of a dataset. Balancing is 
    done to ensure the same resultant distribution for age and sex in 
    case and control.

    Parameters
    ----------
    data : Pandas.DataFrame
        CSV data file
    label_col : str
        Column in data that corresponds to the label to balance for

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
    for ca in case_age.keys():
        a, s = float(ca.split(',')[0]), float(ca.split(',')[1])
        tmp_df = data.loc[(data[label_col] == 0) & 
                            (data['f.21003.0.0'] == a) & 
                            (data['f.31.0.0'] == s)]
        if len(tmp_df) == 0:
            continue
        tmp_df = tmp_df.sample(case_age[ca], replace=False, random_state=7639)
        conts = conts.append(tmp_df)
        del(tmp_df)

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
   
    pgen_prefix = f'{sys_params["RAW_BASE_FOLDER"][chrom]}/withWithdrawn_UKB_chr{chrom}'
    train_ids = pd.read_csv(sys_params["TRAIN_IDS_PATH"], 
                            dtype={'iid':str})['iid'].to_list()
    test_ids = pd.read_csv(sys_params["TEST_IDS_PATH"],
                           dtype={'iid':str})['iid'].to_list()
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
        data_cols = df.columns.to_list()
        num_snps = len(data_cols) - len(covs) - 1
        snp_win = 50
        num_win = int(np.ceil(num_snps/snp_win))
        remaining_snps = num_snps
        # sample_win = np.random.choice(np.arange(0, num_win), 1)
        # sample_win = gsplit[gname]
        for win in range(num_win):
            sind = win * snp_win
            eind = sind+remaining_snps if remaining_snps < snp_win else (win+1)*snp_win
            nsnps = eind-sind
            
            split_f = gene_file.split('_')
            split_f[1] = f'{split_f[1]}_{win}'
            f_win = f'{write_base}/{"_".join(split_f)}'

            cols_win = data_cols[sind+1:eind+1] + covs + [label,]
            df_win = df[cols_win].copy()
            
            # if win == sample_win:
            #     df_win.to_csv(f_win, index=False)
            df_win.to_csv(f_win)

            remaining_snps = remaining_snps - nsnps
