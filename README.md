# Test-Negative Design Inference for Immune Correlates of Protection

This repository evaluates methods for inferring immune correlates of protection using test-negative designs (TNDs).

## Description

This analysis evaluates statistical methods used to infer protection in TNDs focused on immunological assays. 

The general process for this evaluation involves (1) simulating TND data with a known protection function, (2) using an inference pipeline to infer 
protection from simulated data, and (3) comparing the known and inferred protection function. 

## Repository Contents

- **`data_generating_functions.py`** – Generates TND data with a known protection function.  
- **`data_fitting_functions.py`** – Implements logistic regression and the scaled logit model to infer protection from TND data.  
- **`prettyplotlib.py`** – Provides support functions for figure generation.  
- **`generate_figures.ipynb`** – Generates manuscript figures.  
