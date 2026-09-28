# Model Comparison — Logistic Regression vs Gradient Boosting

## 1. Overview

This document compares **Model 1: Logistic Regression** and **Model 2: Gradient Boosting** for the research topic:

> **A Machine Learning Approach to Predicting Retail Investor Dormancy on Digital Investment Platforms**

The comparison evaluates both models on the **same stratified test observations** and combines descriptive performance metrics with paired statistical testing.

The purpose is to understand the different modelling trade-offs rather than relying on accuracy alone.

---

## 2. Related Notebook

The calculations and comparison presented in this document are implemented in:

**`07_Comparison_Logistic_vs_Gradient_Boosting_Annotated(1).ipynb`**

The notebook:

1. Loads the modelling dataset.
2. Creates the same 80/20 stratified train/test split used for both models.
3. Recreates Logistic Regression and Gradient Boosting.
4. Calculates classification and probability-quality metrics.
5. Generates comparison visualisations.
6. Performs McNemar's test.
7. Calculates paired bootstrap confidence intervals for F1-score and ROC-AUC.
8. Saves the comparison results to the `outputs/` directory.

---

## 3. Input Data

The notebook reads:

```text
outputs/dormancy_modelling_dataset.csv
```

The target variable is:

```text
dormant
```

`customer_id` is excluded from the model features.

The data is divided using:

- **Test size:** 20%
- **Training size:** 80%
- **Stratification:** Yes
- **Random state:** 42
- **Classification threshold:** 0.50

Using the same test observations allows the two models to be compared on a paired basis.

---

## 4. Models Compared

### Model 1 — Logistic Regression

The Logistic Regression pipeline uses:

- Median imputation for numerical variables
- StandardScaler for numerical variables
- Most-frequent imputation for categorical variables
- One-hot encoding for categorical variables
- `class_weight="balanced"`
- `C=1.0`
- `solver="liblinear"`
- `max_iter=1000`
- `random_state=42`

The model produces dormancy probabilities using `predict_proba()` and converts them to class predictions using a **0.50 threshold**.

### Model 2 — Gradient Boosting

The Gradient Boosting pipeline uses:

- Median imputation for numerical variables
- Most-frequent imputation for categorical variables
- One-hot encoding for categorical variables
- Balanced sample weights based on the training-set class distribution

Gradient Boosting parameters:

```text
n_estimators = 200
learning_rate = 0.05
max_depth = 3
min_samples_leaf = 20
subsample = 0.80
random_state = 42
```

Predictions are also converted using a **0.50 threshold**.

---

## 5. Performance Metrics

The comparison calculates the following metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct classifications |
| Balanced Accuracy | Average recall across the two classes |
| Precision | Proportion of predicted dormant investors who are actually dormant |
| Recall | Proportion of dormant investors correctly identified |
| Specificity | Proportion of non-dormant investors correctly identified |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Overall class discrimination across thresholds |
| PR-AUC | Positive-class performance across precision/recall thresholds |
| MCC | Balanced classification measure that accounts for all four confusion-matrix categories |
| Log Loss | Quality of predicted probabilities |
| Brier Score | Accuracy of probabilistic predictions |
| TP | True positives |
| TN | True negatives |
| FP | False positives |
| FN | False negatives |

Because dormancy is a minority class, the comparison places particular emphasis on **Recall, Precision, F1-score, PR-AUC, Specificity and probability-quality metrics**.

---

## 6. Observed Performance Results

The notebook reports the following key results on the test set:

| Metric | Logistic Regression | Gradient Boosting |
|---|---:|---:|
| Recall | 0.8590 | 0.8077 |
| Specificity | 0.6127 | 0.7706 |
| F1-score | 0.0669 | 0.1015 |
| ROC-AUC | 0.7866 | 0.8624 |
| PR-AUC | 0.0397 | 0.0671 |
| MCC | 0.1211 | 0.1703 |
| Log Loss | 0.5248 | 0.3768 |
| Brier Score | 0.1932 | 0.1363 |
| False Positives | 3,714 | 2,200 |

### Interpretation

The models show different performance profiles.

**Logistic Regression** has higher recall:

```text
0.8590
```

This means it identifies a larger proportion of the observed dormant investors in this test set. However, it also produces substantially more false-positive alerts and has lower specificity.

**Gradient Boosting** has higher specificity:

```text
0.7706
```

compared with:

```text
0.6127
```

for Logistic Regression. This means it correctly identifies more non-dormant investors and produces fewer false-positive dormancy alerts.

Gradient Boosting also has higher observed values for:

- F1-score
- ROC-AUC
- PR-AUC
- MCC

It also has lower log loss and Brier score, indicating better probability estimates on this test set.

The very low precision observed for both models is consistent with the strong class imbalance in the target. Consequently, accuracy alone would not provide a sufficient description of model behaviour.

---

## 7. False-Positive and False-Negative Trade-off

The models represent different operational trade-offs.

### Logistic Regression

The higher recall means more observed dormant investors are detected.

However, the model produces:

```text
3,714 false positives
```

This means more non-dormant investors would be flagged for potential dormancy intervention.

### Gradient Boosting

Gradient Boosting produces:

```text
2,200 false positives
```

This is substantially fewer than Logistic Regression.

Its recall is lower than Logistic Regression, meaning some dormant investors detected by Logistic Regression are not detected by Gradient Boosting at the 0.50 threshold.

Therefore, the comparison should consider the relative business cost of:

- Missing a dormant investor
- Contacting or intervening with a non-dormant investor unnecessarily

---

## 8. ROC and Precision-Recall Comparison

The notebook generates:

```text
outputs/final_roc_comparison.png
outputs/final_precision_recall_comparison.png
```

### ROC-AUC

The observed ROC-AUC values are:

- Logistic Regression: **0.7866**
- Gradient Boosting: **0.8624**

The higher Gradient Boosting ROC-AUC indicates stronger class discrimination across thresholds on this test set.

### PR-AUC

The observed PR-AUC values are:

- Logistic Regression: **0.0397**
- Gradient Boosting: **0.0671**

Because dormancy is the minority class, the Precision-Recall curve is particularly informative. It focuses attention on the model's ability to identify the positive class rather than allowing the large non-dormant class to dominate the assessment.

---

## 9. McNemar's Test

McNemar's test is used because both models make predictions on the **same test observations**.

It evaluates whether the two models have different paired classification-error patterns.

### Paired Correctness Table

The notebook reports:

| Outcome | Count |
|---|---:|
| Both models correct | 5,847 |
| Logistic Regression correct / Gradient Boosting incorrect | 162 |
| Logistic Regression incorrect / Gradient Boosting correct | 1,668 |
| Both models incorrect | 2,068 |

The McNemar test produces a p-value displayed as:

```text
0.0000
```

at the notebook's displayed precision.

This provides statistical evidence that the models have different paired classification-error patterns on these test observations.

Importantly, McNemar's test does **not** establish that one model is universally better. It tests whether the paired classification outcomes differ.

---

## 10. Paired Bootstrap Confidence Intervals

The notebook also performs a paired bootstrap analysis using:

```text
B = 2000
```

resamples.

The same resampled test observations are used for both models.

The analysis estimates confidence intervals for:

- F1-score difference
- ROC-AUC difference

The reported differences are calculated as:

```text
Gradient Boosting − Logistic Regression
```

### F1-score

Observed difference:

```text
0.0346
```

95% confidence interval:

```text
[0.0263, 0.0428]
```

### ROC-AUC

Observed difference:

```text
0.0759
```

95% confidence interval:

```text
[0.0537, 0.0972]
```

Both reported intervals are above zero in this resampling analysis.

These intervals quantify uncertainty around the observed test-set differences. They should be interpreted as evidence from this test sample rather than as a guarantee that the same differences will occur on every future dataset or population.

The bootstrap results are saved to:

```text
outputs/paired_bootstrap_comparison.csv
```

---

## 11. Statistical Methods Used

The comparison uses two statistical approaches in addition to standard model metrics.

### McNemar's Test

McNemar's test evaluates paired binary classification outcomes.

It is appropriate here because:

- Both models use the same test observations.
- Each observation receives a prediction from both models.
- The analysis focuses on whether their classification errors differ.

### Paired Bootstrap

A paired bootstrap is used to estimate uncertainty around differences in:

- F1-score
- ROC-AUC

The same resampled observations are used for both models, preserving the paired structure of the comparison.

---

## 12. Practical Interpretation

For a retail investment platform, dormancy prediction may support retention or re-engagement activity.

The main metrics have different operational meanings:

- **Recall:** how many potentially dormant investors are detected.
- **Precision:** how targeted the dormancy alerts are.
- **F1-score:** balance between recall and precision.
- **Specificity:** how well the model avoids unnecessary alerts for non-dormant investors.
- **PR-AUC:** useful for evaluating performance on the minority dormant class.
- **ROC-AUC:** overall discrimination between dormant and non-dormant investors.
- **MCC:** balanced classification assessment under class imbalance.
- **Brier Score:** quality of predicted probabilities.
- **Log Loss:** penalises incorrect or poorly calibrated probability estimates.

A higher-recall operating point may be useful where missing a dormant investor is particularly costly. Conversely, stronger specificity may be important where unnecessary customer interventions have a significant operational cost.

The appropriate model and threshold should therefore be connected to the platform's intervention strategy and the relative costs of false positives and false negatives.

---

## 13. Output Files

The comparison notebook produces:

```text
outputs/final_model_comparison.csv
outputs/final_model_performance_comparison.png
outputs/final_roc_comparison.png
outputs/final_precision_recall_comparison.png
outputs/paired_bootstrap_comparison.csv
```

These files contain the numerical comparison, visual performance comparison, ROC/PR curves, and bootstrap results.

---

## 14. How to Run the Comparison

### Prerequisites

The notebook requires Python packages including:

```text
numpy
pandas
matplotlib
scikit-learn
scipy
statsmodels
```

### Steps

1. Ensure the feature-engineered dataset exists:

```text
outputs/dormancy_modelling_dataset.csv
```

2. Open:

```text
07_Comparison_Logistic_vs_Gradient_Boosting_Annotated(1).ipynb
```

3. Run the notebook from top to bottom.

4. The notebook will recreate both models using the same train/test split.

5. Review the displayed performance comparison and statistical results.

6. Check the generated files in:

```text
outputs/
```

---

## 15. Reproducibility

The comparison uses:

```text
RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.50
```

The Gradient Boosting model also uses `random_state=42`.

The paired bootstrap uses a NumPy random generator with `random_state=42` and performs 2,000 resamples.

These settings support reproducibility when the same input dataset and software environment are used.

---

## 16. Summary

The comparison demonstrates two distinct modelling profiles on the same test observations.

**Logistic Regression:**

- Recall: **0.8590**
- Specificity: **0.6127**
- F1-score: **0.0669**
- ROC-AUC: **0.7866**
- PR-AUC: **0.0397**

**Gradient Boosting:**

- Recall: **0.8077**
- Specificity: **0.7706**
- F1-score: **0.1015**
- ROC-AUC: **0.8624**
- PR-AUC: **0.0671**

Logistic Regression identifies a larger proportion of the observed dormant investors at the selected threshold, while Gradient Boosting produces fewer false positives and shows stronger observed discrimination and probability-quality metrics.

McNemar's test indicates different paired error patterns between the models, while the paired bootstrap reports positive observed differences in F1-score and ROC-AUC for Gradient Boosting relative to Logistic Regression.

The comparison therefore provides evidence about the models' different behaviour without treating a single metric as sufficient for selecting an operating model. The final operating point should be aligned with the practical cost of missed dormancy versus unnecessary intervention.
