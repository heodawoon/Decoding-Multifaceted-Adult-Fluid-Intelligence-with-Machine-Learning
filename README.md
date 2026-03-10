# Decoding Multifaceted Cognitive Performance with Machine Learning

This repository contains the code used in the paper:

*Decoding Multifaceted Cognitive Performance with Machine Learning*.

The repository provides the full pipeline used in the study, including data preprocessing, feature construction, model training, evaluation, and feature attribution analyses.

---

# Repository Structure

- **data_preprocessing** – scripts for preparing UK Biobank variables
- **feature_engineering** – construction of social, health, and brain feature sets
- **models** – machine learning and deep learning model implementations
- **training** – training scripts and experiment pipelines
- **evaluation** – performance evaluation and statistical analysis
- **explainability** – SHAP and attribution analyses
- **configs** – configuration files for experiments

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

pip install -r requirements.txt

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

- data preprocessing
- feature construction
- model training
- cross-validation
- evaluation
- feature attribution analyses

Due to UK Biobank access restrictions, users must obtain the dataset independently
and place the processed data in the appropriate directory before running the pipeline.

---

# Citation

If you use this code in your research, please cite:

Heo D.-W., Kim E., Shin E.K., Suk H.-I.  
Decoding Multifaceted Cognitive Performance with Machine Learning.

---

# Disclaimer

This repository is provided to support reproducibility of the analyses reported in the associated research paper.

