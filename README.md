# Mental Health in Tech: A Post-COVID Data Mining Analysis
**Team 3:** Anbreya Busby, Kylie Green, Madison Harris

## Overview
This project analyzes OSMI Mental Health in Tech Survey data from 2016 to 2023 to investigate whether mental health behaviors and workplace support changed following the COVID-19 pandemic shutdowns. Five models are implemented including three classifiers, one clustering model, and one pattern mining model.

## Requirements
pip install pandas numpy scikit-learn mlxtend matplotlib pydunn tabulate

## File Structure
* `preprocess.py` - loads and merges all survey CSV files, assigns survey year
* `clean.py` - cleaning and encoding functions for all variables
* `modeling.py` - all 5 model implementations
* `evaluation.py` - evaluation metrics for all models
* `visualization.py` - generates and saves all charts to the outputs folder
* `main.py` - runs the full pipeline end to end
* `pydunn.py` - Dunn index implementation for clustering evaluation
* `data/` - folder containing all OSMI survey CSV files
* `outputs/` - folder where all charts are saved (created automatically)
* `columns/` - column mapping reference files

## Dataset
OSMI Mental Health in Tech Survey data. Download from https://osmihelp.org/research.html and place all CSV files in the `data/` folder.

## Models
* Random Forest -- predicts respondent era (pre vs post pandemic)
* AdaBoost -- predicts respondent era (pre vs post pandemic)
* Decision Tree -- predicts respondent era (pre vs post pandemic)
* KMeans -- groups respondents into natural clusters
* Association Rule Mining -- finds co-occurring behavioral patterns

## Output
All 8 charts are saved as PNG files to the `outputs/` folder automatically when `main.py` is run.
