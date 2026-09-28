# Model 1 Performance — Logistic Regression

## 1. Overview

**Model 1:** Logistic Regression

**Research topic:** *A Machine Learning Approach to Predicting Retail Investor Dormancy on Digital Investment Platforms*

This document describes the performance evaluation of Model 1 using the dedicated performance notebook:

`05_Performance_Logistic_Regression.ipynb`

The notebook evaluates Logistic Regression using classification, discrimination, probability-quality and error-based measures.

Recall is particularly relevant to the research problem because a false negative represents a dormant investor who was not identified by the model.

---

## 2. Performance Calculation Notebook

All Model 1 performance calculations are contained in:

`05_Performance_Logistic_Regression.ipynb`

The notebook:

1. Loads the feature-engineered modelling dataset.
2. Separates the predictors from the `dormant` target.
3. Excludes `customer_id` from the predictors.
4. Creates a stratified 80/20 train-test split.
5. Applies numerical and categorical preprocessing.
6. Trains the Logistic Regression model.
7. Generates class predictions and predicted probabilities.
8. Calculates multiple performance and statistical measures.
9. Generates a confusion matrix.
10. Generates ROC and Precision-Recall curves.
11. Generates a calibration curve.
12. Saves the performance results and visualisations to the `outputs/` directory.

---

## 3. Input Dataset

The notebook expects the following modelling dataset:

```text
outputs/dormancy_modelling_dataset.csv
```

The target variable is:

```text
dormant
```

The predictor dataset is created using:

```python
X = df.drop(columns=["customer_id", "dormant"])
y = df["dormant"]
```

Therefore:

- `customer_id` is excluded because it is an identifier.
- `dormant` is excluded from the predictors because it is the target variable.

---

## 4. Train-Test Split

The notebook uses a stratified train-test split with the following configuration:

| Setting | Value |
|---|---:|
| Training data | 80% |
| Test data | 20% |
| `TEST_SIZE` | `0.20` |
| `RANDOM_STATE` | `42` |
| Stratification | Yes |

The split is performed using:

```python
train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)
```

Stratification preserves the distribution of the `dormant` classes between the training and test datasets.

The performance metrics are calculated on the held-out test dataset.

---

## 5. Preprocessing

The performance notebook applies separate preprocessing pipelines to numerical and categorical variables.

### Numerical variables

Numerical features are processed using:

1. Median imputation for missing values.
2. Standardisation using `StandardScaler`.

```python
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
```

### Categorical variables

Categorical features are processed using:

1. Most-frequent-value imputation.
2. One-hot encoding.

```python
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
```

The preprocessing is incorporated into the Logistic Regression pipeline.

---

## 6. Logistic Regression Configuration

Model 1 uses the following configuration:

| Parameter | Value |
|---|---:|
| Model | Logistic Regression |
| `C` | `1.0` |
| `class_weight` | `balanced` |
| `solver` | `liblinear` |
| `max_iter` | `1000` |
| `random_state` | `42` |
| Classification threshold | `0.50` |

The model is defined as:

```python
LogisticRegression(
    C=1.0,
    class_weight="balanced",
    solver="liblinear",
    max_iter=1000,
    random_state=RANDOM_STATE
)
```

`class_weight="balanced"` is used to account for class imbalance during model training.

---

## 7. Predictions

The notebook generates predicted probabilities for the positive class:

```python
prob = model.predict_proba(X_test)[:,1]
```

The final class prediction is generated using a threshold of 0.50:

```python
pred = (prob >= THRESHOLD).astype(int)
```

Therefore:

- Probability ≥ 0.50 → predicted `1` / Dormant
- Probability < 0.50 → predicted `0` / Non-dormant

The predicted probabilities are also used for ROC-AUC, PR-AUC, Log Loss, Brier Score and the calibration analysis.

---

## 8. Performance Metrics

The notebook calculates a broader set of performance measures than simple accuracy.

### 8.1 Accuracy

Accuracy measures the proportion of all test observations classified correctly.

```text
Accuracy =
(TP + TN) / (TP + TN + FP + FN)
```

---

### 8.2 Balanced Accuracy

Balanced Accuracy accounts for both classes and is calculated as the average of sensitivity and specificity.

It is useful when the target classes are imbalanced.

---

### 8.3 Precision

Precision measures the proportion of investors predicted as dormant who were actually dormant.

```text
Precision =
TP / (TP + FP)
```

---

### 8.4 Recall

Recall measures the proportion of actual dormant investors correctly identified by the model.

```text
Recall =
TP / (TP + FN)
```

For this research problem, recall is particularly important because false negatives represent dormant investors who were not identified for possible retention activity.

---

### 8.5 Specificity

Specificity measures the model's ability to correctly identify non-dormant investors.

```text
Specificity =
TN / (TN + FP)
```

A lower specificity means that more non-dormant investors are incorrectly flagged as dormant.

---

### 8.6 F1-score

F1-score balances precision and recall using their harmonic mean.

```text
F1 =
2 × (Precision × Recall) / (Precision + Recall)
```

---

### 8.7 ROC-AUC

ROC-AUC measures the model's ability to discriminate between dormant and non-dormant investors across classification thresholds.

The calculation uses predicted probabilities rather than only the final 0/1 predictions.

---

### 8.8 PR-AUC

PR-AUC represents the Average Precision score.

It evaluates the relationship between precision and recall across classification thresholds.

This measure is particularly useful when dormancy is a minority class.

---

### 8.9 Matthews Correlation Coefficient (MCC)

MCC measures the quality of binary classifications using all four confusion-matrix outcomes:

- True positives
- True negatives
- False positives
- False negatives

It is useful as a balanced measure when the classes are unevenly distributed.

---

### 8.10 Log Loss

Log Loss evaluates the quality of the model's predicted probabilities.

It penalises predictions that are confidently incorrect.

The notebook calculates it using:

```python
log_loss(y_true, prob)
```

---

### 8.11 Brier Score

The Brier Score measures the accuracy of probabilistic predictions.

It compares predicted probabilities with the actual binary outcomes.

Lower values indicate better probabilistic accuracy.

---

## 9. Confusion Matrix

The notebook calculates:

- True Positives (`TP`)
- True Negatives (`TN`)
- False Positives (`FP`)
- False Negatives (`FN`)

using:

```python
confusion_matrix(
    y_true,
    pred,
    labels=[0,1]
)
```

The resulting confusion matrix is displayed using the labels:

- Non-dormant
- Dormant

The output is saved as:

```text
outputs/lr_performance_confusion_matrix.png
```

The figure is saved at 300 DPI.

---

## 10. ROC Curve

The notebook generates a Receiver Operating Characteristic (ROC) curve using the actual test labels and predicted probabilities.

The curve evaluates model discrimination across different classification thresholds.

The output is saved as:

```text
outputs/lr_performance_roc_curve.png
```

The figure is saved at 300 DPI.

The corresponding ROC-AUC value is also included in the performance results.

---

## 11. Precision-Recall Curve

The notebook generates a Precision-Recall curve using:

```python
PrecisionRecallDisplay.from_predictions(
    y_test,
    prob,
    name="Logistic Regression"
)
```

The Precision-Recall curve is particularly relevant when the positive class is relatively uncommon because it focuses directly on the relationship between precision and recall.

The output is saved as:

```text
outputs/lr_performance_precision_recall_curve.png
```

The figure is saved at 300 DPI.

---

## 12. Calibration Curve

The notebook evaluates the quality of the model's predicted probabilities using a calibration curve.

The calculation uses:

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

The notebook also displays a reference line representing perfect calibration.

The resulting figure is saved as:

```text
outputs/lr_performance_calibration_curve.png
```

The figure is saved at 300 DPI.

---

## 13. Model 1 Performance Results

The notebook calculates the following results:

| Metric | Logistic Regression |
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

The notebook saves the calculated values to:

```text
outputs/logistic_regression_performance.csv
```

The supplied performance notebook contains the code that calculates these values. If the notebook is not executed with the required modelling dataset, numeric results should not be manually estimated or invented.

---

## 14. Interpretation of Errors

The notebook frames the model's errors in terms of the research problem.

### False negatives

A false negative represents an investor who becomes dormant but is not identified by the model.

These cases could represent missed opportunities for possible retention activity.

### False positives

A false positive represents a non-dormant investor who is incorrectly identified as dormant.

These predictions could result in unnecessary or poorly targeted interventions.

The model's practical impact therefore needs to be considered alongside the operational costs associated with false positives and false negatives.

---

## 15. Output Files

After successful execution, the performance notebook generates:

```text
outputs/
├── logistic_regression_performance.csv
├── lr_performance_confusion_matrix.png
├── lr_performance_roc_curve.png
├── lr_performance_precision_recall_curve.png
└── lr_performance_calibration_curve.png
```

| Output file | Purpose |
|---|---|
| `logistic_regression_performance.csv` | Stores all calculated performance metrics and confusion-matrix counts |
| `lr_performance_confusion_matrix.png` | Visualises TP, TN, FP and FN |
| `lr_performance_roc_curve.png` | Shows ROC performance across thresholds |
| `lr_performance_precision_recall_curve.png` | Shows the precision-recall relationship |
| `lr_performance_calibration_curve.png` | Assesses probability calibration |

---

## 16. How to Run the Performance Notebook

### Step 1 — Prepare the modelling dataset

Ensure the feature-engineered dataset exists at:

```text
outputs/dormancy_modelling_dataset.csv
```

### Step 2 — Open the notebook

Open:

```text
05_Performance_Logistic_Regression.ipynb
```

### Step 3 — Run all cells

Run the notebook cells sequentially from top to bottom.

The notebook will:

1. Load the modelling dataset.
2. Create the stratified train-test split.
3. Train Model 1.
4. Generate predictions and probabilities.
5. Calculate the performance metrics.
6. Generate the evaluation plots.
7. Save the results to the `outputs/` directory.

### Required Python packages

The notebook uses:

- NumPy
- Pandas
- Matplotlib
- scikit-learn
- SciPy
- statsmodels

If required, install them with:

```bash
pip install numpy pandas matplotlib scikit-learn scipy statsmodels
```

---

## 17. Reproducibility

The performance notebook uses:

```text
Random state = 42
Test size = 0.20
Classification threshold = 0.50
Logistic Regression C = 1.0
Class weight = balanced
Solver = liblinear
Maximum iterations = 1000
Calibration bins = 10
Calibration strategy = quantile
```

Using the same modelling dataset and configuration should reproduce the same train-test split and performance calculations.

---

## 18. Summary

Model 1 is evaluated using a dedicated performance notebook:

`05_Performance_Logistic_Regression.ipynb`

The evaluation goes beyond basic classification accuracy by examining:

- Classification performance
- Class-balanced performance
- Discrimination
- Precision-recall behaviour
- Probability quality
- Classification errors
- Calibration

The main performance measures are Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1-score, ROC-AUC, PR-AUC, MCC, Log Loss and Brier Score.

The evaluation is supported by a confusion matrix, ROC curve, Precision-Recall curve and calibration curve.

The complete performance analysis can be reproduced by running:

`05_Performance_Logistic_Regression.ipynb`
