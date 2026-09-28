# Gradient Boosting Performance — Model 2

## 1. Overview

**Model 2:** Gradient Boosting

**Research topic:** *A Machine Learning Approach to Predicting Retail Investor Dormancy on Digital Investment Platforms*

This document describes the performance evaluation of Model 2 using the dedicated performance notebook:

`06_Performance_Gradient_Boosting.ipynb`

The notebook evaluates Gradient Boosting using the same held-out test-set framework used for Logistic Regression.

---

## 2. Performance Calculation Notebook

The complete Model 2 performance calculations are contained in:

`06_Performance_Gradient_Boosting.ipynb`

The notebook:

1. Loads the feature-engineered modelling dataset.
2. Separates the predictors from the `dormant` target.
3. Excludes `customer_id` from the predictors.
4. Creates a stratified 80/20 train-test split.
5. Applies the required numerical and categorical preprocessing.
6. Trains the Gradient Boosting classifier.
7. Applies balanced sample weights during training.
8. Generates class predictions and predicted probabilities.
9. Calculates the performance metrics.
10. Generates a confusion matrix.
11. Generates ROC and Precision-Recall curves.
12. Generates a calibration curve.
13. Saves the performance results and visualisations to the `outputs/` directory.

---

## 3. Input Dataset

The notebook expects:

```text
outputs/dormancy_modelling_dataset.csv
```

The target variable is:

```text
dormant
```

The predictor matrix is created by excluding:

```text
customer_id
dormant
```

`customer_id` is excluded because it is an identifier rather than a predictive feature.

---

## 4. Train-Test Split

The notebook uses a stratified train-test split:

| Setting | Value |
|---|---:|
| Training data | 80% |
| Test data | 20% |
| `TEST_SIZE` | `0.20` |
| `RANDOM_STATE` | `42` |
| Stratification | Yes |

The split is performed using the `dormant` target to preserve the target-class distribution between the training and test sets.

Performance metrics are calculated using the held-out test set.

---

## 5. Preprocessing

The notebook uses separate preprocessing pipelines for numerical and categorical variables.

### Numerical variables

Missing numerical values are replaced using the median:

```python
SimpleImputer(strategy="median")
```

No numerical standardisation is applied in this performance notebook.

### Categorical variables

Categorical variables are:

1. Imputed using the most frequent value.
2. One-hot encoded.

```python
OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)
```

`handle_unknown="ignore"` prevents unseen categorical values in the test set from causing an encoding error.

---

## 6. Class Imbalance Handling

Balanced sample weights are calculated from the training data.

The notebook calculates:

```python
positive = y_train.sum()
negative = len(y_train) - positive

weights = np.where(
    y_train == 1,
    len(y_train) / (2 * positive),
    len(y_train) / (2 * negative)
)
```

These weights are supplied when fitting the Gradient Boosting model:

```python
model.fit(
    X_train_gb,
    y_train,
    sample_weight=weights
)
```

The class balancing is therefore applied during model training rather than by modifying the held-out test set.

---

## 7. Gradient Boosting Configuration

Model 2 uses the following configuration:

| Parameter | Value |
|---|---:|
| Model | `GradientBoostingClassifier` |
| `n_estimators` | `200` |
| `learning_rate` | `0.05` |
| `max_depth` | `3` |
| `min_samples_leaf` | `20` |
| `subsample` | `0.80` |
| `random_state` | `42` |
| Classification threshold | `0.50` |

The model is defined as:

```python
GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=20,
    subsample=0.80,
    random_state=RANDOM_STATE
)
```

---

## 8. Predictions

The model generates predicted probabilities for the positive class:

```python
prob = model.predict_proba(X_test_gb)[:, 1]
```

The final class prediction uses a threshold of 0.50:

```python
pred = (prob >= THRESHOLD).astype(int)
```

Therefore:

- Probability ≥ 0.50 → predicted `1` / Dormant
- Probability < 0.50 → predicted `0` / Non-dormant

The predicted probabilities are also used for ROC-AUC, PR-AUC, Log Loss, Brier Score and calibration analysis.

---

## 9. Performance Metrics

The notebook calculates the following metrics:

### Accuracy

Measures the proportion of all test observations classified correctly.

### Balanced Accuracy

Measures classification performance while accounting for both target classes.

### Precision

Measures the proportion of predicted dormant investors who are actually dormant.

### Recall

Measures the proportion of actual dormant investors correctly identified by the model.

### Specificity

Measures the proportion of non-dormant investors correctly identified as non-dormant.

### F1-score

Combines precision and recall using their harmonic mean.

### ROC-AUC

Measures the model's ability to distinguish between dormant and non-dormant investors across classification thresholds.

### PR-AUC

Measures the precision-recall performance across thresholds. This is particularly relevant when the dormant class is less common.

### Matthews Correlation Coefficient (MCC)

Uses all four confusion-matrix outcomes to measure binary classification quality.

### Log Loss

Measures the quality of the predicted probabilities and penalises confident incorrect predictions.

### Brier Score

Measures the accuracy of probabilistic predictions.

---

## 10. Model 2 Performance Results

The notebook calculates:

| Metric | Gradient Boosting |
|---|---:|
| Accuracy | Generated by notebook |
| Balanced Accuracy | Generated by notebook |
| Precision | Generated by notebook |
| Recall | Generated by notebook |
| Specificity | Generated by notebook |
| F1-score | Generated by notebook |
| ROC-AUC | Generated by notebook |
| PR-AUC | Generated by notebook |
| MCC | Generated by notebook |
| Log Loss | Generated by notebook |
| Brier Score | Generated by notebook |
| TP | Generated by notebook |
| TN | Generated by notebook |
| FP | Generated by notebook |
| FN | Generated by notebook |

The actual results are saved to:

```text
outputs/gradient_boosting_performance.csv
```

The numeric values should be obtained by executing the notebook with the required modelling dataset. They should not be manually estimated.

---

## 11. Confusion Matrix

The notebook generates a confusion matrix using:

```python
ConfusionMatrixDisplay.from_predictions(
    y_test,
    pred,
    display_labels=["Non-dormant", "Dormant"]
)
```

The confusion matrix identifies:

| Outcome | Meaning |
|---|---|
| True Positive | Dormant investor correctly identified |
| True Negative | Non-dormant investor correctly identified |
| False Positive | Non-dormant investor incorrectly identified as dormant |
| False Negative | Dormant investor incorrectly missed |

The output is saved as:

```text
outputs/gb_performance_confusion_matrix.png
```

The image is saved at 300 DPI.

---

## 12. ROC Curve

The notebook generates a Receiver Operating Characteristic (ROC) curve from the test labels and predicted probabilities.

The ROC curve evaluates model discrimination across classification thresholds.

The output is saved as:

```text
outputs/gb_performance_roc_curve.png
```

The image is saved at 300 DPI.

---

## 13. Precision-Recall Curve

The notebook generates a Precision-Recall curve using the predicted probabilities.

This provides a view of the trade-off between:

- Precision
- Recall

The PR curve is particularly useful when the dormant class may be less common than the non-dormant class.

The output is saved as:

```text
outputs/gb_performance_precision_recall_curve.png
```

The image is saved at 300 DPI.

---

## 14. Calibration Curve

The notebook evaluates probability calibration using:

```python
calibration_curve(
    y_test,
    prob,
    n_bins=10,
    strategy="quantile"
)
```

The calibration curve compares:

- Mean predicted probability
- Observed proportion of positive outcomes

A perfect-calibration reference line is also plotted.

The output is saved as:

```text
outputs/gb_performance_calibration_curve.png
```

The image is saved at 300 DPI.

---

## 15. Interpretation and Practical Impact

Gradient Boosting can model nonlinear relationships and interactions between investor characteristics and historical transaction behaviour.

Recall indicates how effectively the model identifies investors who become dormant, while precision indicates the reliability of the resulting dormancy alerts.

PR-AUC is relevant because the dormant class may be less common than the non-dormant class.

Probability-based measures such as Brier Score and Log Loss are useful when predicted dormancy probabilities are used to prioritise investors according to risk rather than relying only on a binary classification.

The practical impact should be considered through false negatives and false positives:

- **False negatives:** dormant investors that are missed, potentially reducing the reach of retention efforts.
- **False positives:** non-dormant investors incorrectly flagged as dormant, potentially increasing intervention costs or unnecessary customer contact.

---

## 16. Output Files

After successful execution, the notebook generates:

```text
outputs/
├── gradient_boosting_performance.csv
├── gb_performance_confusion_matrix.png
├── gb_performance_roc_curve.png
├── gb_performance_precision_recall_curve.png
└── gb_performance_calibration_curve.png
```

| Output | Purpose |
|---|---|
| `gradient_boosting_performance.csv` | Stores the calculated performance metrics and confusion-matrix counts |
| `gb_performance_confusion_matrix.png` | Visualises classification outcomes |
| `gb_performance_roc_curve.png` | Shows ROC performance |
| `gb_performance_precision_recall_curve.png` | Shows precision-recall performance |
| `gb_performance_calibration_curve.png` | Shows probability calibration |

---

## 17. How to Run the Performance Notebook

### Step 1 — Prepare the modelling dataset

Ensure:

```text
outputs/dormancy_modelling_dataset.csv
```

exists.

### Step 2 — Open the notebook

Open:

```text
06_Performance_Gradient_Boosting.ipynb
```

### Step 3 — Run all cells

Run the notebook cells sequentially from top to bottom.

The notebook will:

1. Load the modelling dataset.
2. Create the stratified train-test split.
3. Preprocess the data.
4. Calculate balanced sample weights.
5. Train the Gradient Boosting model.
6. Generate predictions and probabilities.
7. Calculate all performance metrics.
8. Generate the evaluation plots.
9. Save the outputs.

### Required Python packages

The notebook uses:

- NumPy
- Pandas
- Matplotlib
- scikit-learn
- SciPy
- statsmodels

Install them if necessary:

```bash
pip install numpy pandas matplotlib scikit-learn scipy statsmodels
```

---

## 18. Reproducibility

The notebook uses:

```text
Random state = 42
Test size = 0.20
Classification threshold = 0.50
n_estimators = 200
learning_rate = 0.05
max_depth = 3
min_samples_leaf = 20
subsample = 0.80
Calibration bins = 10
Calibration strategy = quantile
```

Using the same modelling dataset and configuration should reproduce the performance calculations.

---

## 19. Summary

Model 2 is a Gradient Boosting classifier developed to predict retail investor dormancy.

The dedicated performance notebook:

`06_Performance_Gradient_Boosting.ipynb`

evaluates the model using classification, discrimination, probability-quality and calibration measures.

The main metrics are:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- Specificity
- F1-score
- ROC-AUC
- PR-AUC
- MCC
- Log Loss
- Brier Score

The evaluation is supported by a confusion matrix, ROC curve, Precision-Recall curve and calibration curve.

The complete performance analysis can be reproduced by running:

`06_Performance_Gradient_Boosting.ipynb`
