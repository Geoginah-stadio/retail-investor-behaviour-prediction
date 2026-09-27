# Retail Investor Dormancy — Performance Evaluation

Run the notebooks after generating `outputs/dormancy_modelling_dataset.csv`.

## Evaluation notebooks

1. `05_Performance_Logistic_Regression.ipynb`
2. `06_Performance_Gradient_Boosting.ipynb`
3. `07_Comparison_Logistic_vs_Gradient_Boosting.ipynb`

The notebooks evaluate accuracy, balanced accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC, MCC, log loss and Brier score. The comparison also includes McNemar's test and paired bootstrap confidence intervals.

## Environment

Create an environment:

`python -m venv .venv`

Windows:
`.venv\Scripts\activate`

macOS/Linux:
`source .venv/bin/activate`

Install:

`pip install -r requirements.txt`

Launch:

`jupyter notebook`
