# Decoding Multifaceted Adult Fluid Intelligence with Machine Learning: Juxtaposing Social, Medical, and Brain Factors using UK Biobank

Code accompanying the manuscript:

Da-Woon Heo†, Eunjae Kim†, Sohyun Kang, Joon-Kyung Seong, Heung-Il Suk*, and Eun Kyong Shin*
*Decoding Multifaceted Adult Fluid Intelligence with Machine Learning:
Juxtaposing Social, Medical, and Brain Factors using UK Biobank*

† Equal contribution; * Corresponding authors

(Manuscript submitted)

This repository provides the full pipeline used in the study, including data preprocessing, feature construction, model training, evaluation, and feature attribution analyses.

---
# Repository Structure

The repository is organized into several components corresponding to the main analysis stages.

### Data preprocessing

`Data_process/`

Scripts used to construct the analysis dataset from the UK Biobank resource.

Pipeline steps include:

- Step0: variable extraction and merging (`Step0_merge_and_extract_complete_fractional_anisotropy_data.py`)
- Step1: variable recoding (`Step1_variable_recoding_and_renaming.py`)
- Step2: disease date alignment relative to imaging visits (`Step2_redefine_disease_dates_and_code_timing_relative_to_imaging.py`)
- Step3: filtering brain-related diseases (`Step3_filter_brain_related_disease.py`)
- Step4: missing-value filtering (`Step4_filter_values.py`)
- Step5: dataset preparation for machine learning and deep learning (`Step5_re_filter_values_for_deeplearning.py`)
- Step6: cross-validation split generation (5 repetitions of 5-fold CV) (`Step6_split_5_repeat_5_fold.py`)

Key scripts:

- `Step4_filter_values.py` – generates `Step4_4_binarize_disease_column.csv` used for statistical analysis in R.
- `Step5_re_filter_values_for_deeplearning.py` – generates `Step5_refilter_categorical_for_deeplearning.csv` used for machine learning and deep learning models.
- `Step6_split_5_repeat_5_fold.py` – generates the cross-validation splits used in all experiments.

---

### Tree-based machine learning models

`Tree_based_models/`

Implementation of tree-based models used in the study:

- XGBoost
- Random Forest
- LightGBM

Includes:

- model training pipelines
- SHAP-based feature attribution analyses
- knee-point based feature selection

Important scripts:

- `xgboost_pipeline_shap.py`
- `random_forest_pipeline_shap.py`
- `lightgbm_pipeline_shap.py`

Feature attribution scripts are located in:

`Tree_based_models/Interpret/`

---

### Deep learning model (FT-Transformer)

`DL_based_model/FT_Transformer/`

Implementation of the deep learning model used in this study.

Components include:

- model architecture
- training pipeline
- configuration files
- utility functions

Key scripts:

- `main.py` – training entry point
- `main_interpret.py` – interpretation pipeline
- `summarize_gradient_shap.py` – attribution summarization

The implementation is based on the FT-Transformer architecture proposed by Gorishniy et al. (NeurIPS 2021).

---

### Statistical analyses

`R_script/`

Contains additional statistical analyses performed in the study, including:

- logistic regression models
- average treatment effect (ATE) analysis

---

# Environment

Python version used in the experiments:

Python 3.10

Main dependencies:

- xgboost
- lightgbm
- scikit-learn
- pytorch
- shap
- captum
- kneed

Install dependencies:

```
pip install -r requirements.txt
```

---

# External Libraries and Frameworks

This project uses several external libraries and frameworks:

### Machine Learning

- XGBoost  
- LightGBM  
- scikit-learn  

### Deep Learning

- PyTorch  

### Explainability

- SHAP  
- Captum  

### Utility Libraries

- kneed (used for knee-point based feature selection)

---

# FT-Transformer Implementation

The deep learning model in this study is based on the **FT-Transformer architecture**
proposed by:

Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data*, NeurIPS 2021.

Official implementation:  
https://github.com/yandex-research/rtdl-revisiting-models

The original architecture and core components were used as a reference,
while the training pipeline and data integration were adapted for the
multi-domain cognitive prediction tasks in this study.

---

# Data Access

This study uses data from the **UK Biobank**.

Due to data access restrictions, the dataset cannot be redistributed in this repository.

Researchers can obtain access through the official UK Biobank data access procedure:

https://www.ukbiobank.ac.uk/

The analyses in this study were conducted under **Application ID 70034**.

---

# Reproducibility

This repository contains all scripts required to reproduce the analyses reported in the paper:

- data preprocessing: `Data_process/`
- feature construction and cross-validation splits: `Data_process/`
- model training with cross-validation: `Tree_based_models/` and `DL_based_model/FT_Transformer/`
- evaluation: `Tree_based_models/` and `DL_based_model/FT_Transformer/`
- feature attribution analyses: `R_script/`

All experiments use the same cross-validation splits generated by
`Step6_split_5_repeat_5_fold.py`.

Due to UK Biobank access restrictions, users must obtain the dataset independently
and place the processed data in the appropriate directory before running the pipeline.

---

# Disclaimer

This repository is provided to support reproducibility of the analyses reported in the associated research paper.

