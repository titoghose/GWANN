# Genome Wide Association Neural Networks (GWANN) 

## Manuscript

[Genome wide association neural networks (GWANN) identify novel genes linked to family history of Alzheimer’s disease in the UK BioBank](https://www.medrxiv.org/content/10.1101/2022.06.10.22276251)


## Analysis parameters

The analysis requires parameters and covariates in a yaml file.

### Parameters

```Yaml
DATA_BASE_FOLDER: <Path to folder where csv data files should be saved>
DATA_LOGS: <Path to folder where data generation logs should be saved>
LOGS_BASE_FOLDER: <Path to folder where training plots and models should be saved>
RUNS_BASE_FOLDER: <Path to folder where training info and logs should be written>
PARAMS_PATH: <Path to this yaml file>
PHEN_COV_PATH: <Path to file with participant IDs, phenotype and covariate information>
RAW_BASE_FOLDER:
  '1': <Path to folder containing pgen file for chromosome>
  '10': <Path to folder containing pgen file for chromosome>
  '11': <Path to folder containing pgen file for chromosome>
  '12': <Path to folder containing pgen file for chromosome>
  '13': <Path to folder containing pgen file for chromosome>
  '14': <Path to folder containing pgen file for chromosome>
  '15': <Path to folder containing pgen file for chromosome>
  '16': <Path to folder containing pgen file for chromosome>
  '17': <Path to folder containing pgen file for chromosome>
  '18': <Path to folder containing pgen file for chromosome>
  '19': <Path to folder containing pgen file for chromosome>
  '2': <Path to folder containing pgen file for chromosome>
  '20': <Path to folder containing pgen file for chromosome>
  '21': <Path to folder containing pgen file for chromosome>
  '22': <Path to folder containing pgen file for chromosome>
  '3': <Path to folder containing pgen file for chromosome>
  '4': <Path to folder containing pgen file for chromosome>
  '5': <Path to folder containing pgen file for chromosome>
  '6': <Path to folder containing pgen file for chromosome>
  '7': <Path to folder containing pgen file for chromosome>
  '8': <Path to folder containing pgen file for chromosome>
  '9': <Path to folder containing pgen file for chromosome>
  'Dummy': <Path to folder containing pgen file with dummy data>
```

### Covariates

List of covariates to use for the analysis. The ones given below were used in the analysis described in the manuscript.

```Yaml
COVARIATES : 
- 'f.31.0.0'
- 'f.21003.0.0'
- 'f.6138_ISCED'
- 'f.22009.0.1'
- 'f.22009.0.2'
- 'f.22009.0.3'
- 'f.22009.0.4'
- 'f.22009.0.5'
- 'f.22009.0.6'
```

## White-British Population

The pipeline in `Code_AD/run_pipeline.sh` was used to run the analysis genome wide.
- It first trains the covariate model
- Then trains the models for each gene window
- Finally, trains models for dummy gene windows

## Non-white Populations

After training the models on the White-British Population, the `Code_AD/non_white_validation.sh` was used to run the models in inference mode on the Asian and Black populations in the UK Biobank.
- It obtains the metrics for all random seeds and both populations for the top 100 gene windows from the white-british population
- Finally, it generates dummy SNPs for each population and obtains the metrics for the dummy windows

## P-value calculations

The script `Code_AD/post_hoc_scripts/calc_pvalue.R` is used to obtain the P-value from a single file containing the metrics of every gene window for all random seeds, and a similar file for all dummy gene windows.

## Post-hoc analyses

The scripts in the folder `Code_AD/post_hoc_scripts` were used to perform the series of post hoc analyses discussed in the manuscript. 