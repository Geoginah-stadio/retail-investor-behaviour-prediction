# Gradient Boosting Model

## 1. Overview

**Model:** Gradient Boosting Classifier

**Research topic:** *A Machine Learning Approach to Predicting Retail Investor Dormancy on Digital Investment Platforms*

This document describes the development and evaluation of the Gradient Boosting model.

The complete implementation is contained in:

`04_Gradient_Boosting.ipynb`

The modelling dataset is produced by the feature-engineering stage and is located at:

`outputs/dormancy_modelling_dataset.csv`

The notebook independently develops and evaluates the Gradient Boosting model, including preprocessing, class-imbalance handling, model training, predictions, performance metrics, a confusion matrix and an ROC curve.

---

## 2. Modelling Dataset

The Gradient Boosting notebook loads:

```text
outputs/dormancy_modelling_dataset.csv
```

The target variable is:

```text
dormant
```

The following columns are excluded from the predictor variables:

- `customer_id`
- `dormant`

Therefore:

```python
X = df.drop(columns=["customer_id", "dormant"])
y = df["dormant"]
```

`customer_id` is excluded because it is an identifier rather than a predictive feature.

---

## 3. Train-Test Split

The data is divided into training and testing datasets using a stratified train-test split.

| Setting | Value |
|---|---:|
| Training data | 80% |
| Testing data | 20% |
| Test size | `0.20` |
| Random state | `42` |
| Stratification | Yes |

Stratification is applied using the `dormant` target so that the class distribution is maintained between the training and testing datasets.

The test dataset is retained in its original class distribution for model evaluation.

---

## 4. Preprocessing

Gradient Boosting is a tree-based algorithm and therefore does **not require numerical standardisation**.

The preprocessing performed by the notebook is:

### Numerical variables

Missing numerical values are replaced using the median.

```python
SimpleImputer(strategy="median")
```

### Categorical variables

Categorical variables are processed using:

1. Most-frequent-value imputation for missing values.
2. One-hot encoding of categorical variables.

The notebook uses:

```python
OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)
```

Using `handle_unknown="ignore"` allows the preprocessing step to handle categorical values that may appear in the test data but were not present during training.

---

## 5. Class Imbalance Handling

The `dormant` outcome is the class of interest, so the model uses balanced sample weights during training.

The training data is used to calculate separate weights for:

- Dormant investors (`dormant = 1`)
- Non-dormant investors (`dormant = 0`)

The notebook calculates the weights as:

```python
sample_weights = np.where(
    y_train == 1,
    len(y_train) / (2 * positive_count),
    len(y_train) / (2 * negative_count)
)
```

This gives greater weight to the minority class when the model is trained.

The test set is **not rebalanced**, because it is intended to represent the original class distribution when evaluating model performance.

---

## 6. Gradient Boosting Model Configuration

The Gradient Boosting Classifier is configured with the following hyperparameters:

| Parameter | Value |
|---|---:|
| Model | `GradientBoostingClassifier` |
| `n_estimators` | 200 |
| `learning_rate` | 0.05 |
| `max_depth` | 3 |
| `min_samples_leaf` | 20 |
| `subsample` | 0.80 |
| `random_state` | 42 |
| Classification threshold | 0.50 |

### Parameter descriptions

**`n_estimators = 200`**

The model uses 200 boosting stages.

**`learning_rate = 0.05`**

Controls the contribution of each boosting stage to the final model.

**`max_depth = 3`**

Controls the maximum depth of the individual decision trees used by the boosting model.

**`min_samples_leaf = 20`**

Requires a minimum of 20 observations in a leaf, helping constrain tree complexity.

**`subsample = 0.80`**

Each boosting stage uses 80% of the available training observations.

**`random_state = 42`**

Provides reproducibility.

---

## 7. Model Training

The model is trained using the processed training data and the calculated balanced sample weights:

```python
gradient_boosting_model.fit(
    X_train_processed,
    y_train,
    sample_weight=sample_weights
)
```

After training, the model generates probabilities for the positive class:

```python
gb_prob = gradient_boosting_model.predict_proba(
    X_test_processed
)[:, 1]
```

The final class prediction uses a threshold of 0.50:

```python
gb_pred = (gb_prob >= THRESHOLD).astype(int)
```

Therefore:

- Probability ≥ 0.50 → `Dormant`
- Probability < 0.50 → `Non-dormant`

---

## 8. Model Evaluation

The Gradient Boosting model is evaluated on the held-out test dataset.

The notebook calculates:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

These metrics are calculated using the test-set predictions and predicted probabilities.

---

## 9. Performance Metrics

### Accuracy

Accuracy measures the proportion of all test observations that were classified correctly.

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Precision

Precision measures the proportion of observations predicted as dormant that were actually dormant.

```text
Precision =
True Positives / (True Positives + False Positives)
```

### Recall

Recall measures the proportion of actual dormant investors that were correctly identified.

```text
Recall =
True Positives / (True Positives + False Negatives)
```

### F1-score

F1-score combines precision and recall using their harmonic mean.

```text
F1 =
2 × (Precision × Recall) / (Precision + Recall)
```

### ROC-AUC

ROC-AUC evaluates the model's ability to distinguish between dormant and non-dormant investors across different classification thresholds.

The ROC-AUC is calculated using the predicted probability of the dormant class:

```python
roc_auc_score(y_test, gb_prob)
```

---

## 10. Gradient Boosting Results

The notebook calculates the following performance results:

| Metric | Gradient Boosting |
|---|---:|
| Accuracy | Generated by notebook |
| Precision | Generated by notebook |
| Recall | Generated by notebook |
| F1-score | Generated by notebook |
| ROC-AUC | Generated by notebook |

The notebook saves the actual calculated values to:

```text
outputs/gradient_boosting_metrics.csv
```

**Note:** The supplied notebook contains the metric-calculation code but does not contain executed numeric metric outputs. The results should therefore be populated by running the notebook rather than manually estimated.

---

## 11. Classification Report

The notebook also produces a classification report for:

- Non-dormant
- Dormant

The report contains:

- Precision
- Recall
- F1-score
- Support

It is generated using:

```python
classification_report(
    y_test,
    gb_pred,
    target_names=["Non-dormant", "Dormant"],
    digits=4,
    zero_division=0
)
```

---

## 12. Confusion Matrix

A confusion matrix is generated from the test-set predictions.

It provides four classification outcomes:

| Outcome | Description |
|---|---|
| True Positive | Dormant investor correctly classified as dormant |
| True Negative | Non-dormant investor correctly classified as non-dormant |
| False Positive | Non-dormant investor incorrectly classified as dormant |
| False Negative | Dormant investor incorrectly classified as non-dormant |

The resulting figure is saved as:

```text
outputs/gradient_boosting_confusion_matrix.png
```

The figure is saved at 300 DPI.

---

## 13. ROC Curve

The notebook generates a Receiver Operating Characteristic (ROC) curve using the predicted probabilities from the Gradient Boosting model.

The ROC curve illustrates the relationship between:

- True Positive Rate
- False Positive Rate

across different classification thresholds.

The resulting figure is saved as:

```text
outputs/gradient_boosting_roc_curve.png
```

The figure is saved at 300 DPI.

---

## 14. Statistical and Evaluation Methods

The Gradient Boosting evaluation uses the following methods:

1. **Stratified train-test split** — 80% training and 20% testing.
2. **Balanced sample weighting** — gives greater training weight to the dormant class.
3. **Accuracy** — measures overall classification correctness.
4. **Precision** — measures the correctness of dormant predictions.
5. **Recall** — measures identification of actual dormant investors.
6. **F1-score** — combines precision and recall.
7. **ROC-AUC** — measures class discrimination using predicted probabilities.
8. **Confusion matrix** — examines the four classification outcomes.
9. **Classification report** — provides class-level precision, recall, F1-score and support.
10. **ROC curve** — evaluates classification performance across thresholds.

The supplied notebook does not implement a separate hypothesis test, confidence interval calculation, or statistical significance test.

---

## 15. Output Files

After successful execution, the notebook produces:

```text
outputs/
├── gradient_boosting_metrics.csv
├── gradient_boosting_confusion_matrix.png
└── gradient_boosting_roc_curve.png
```

| Output | Purpose |
|---|---|
| `gradient_boosting_metrics.csv` | Stores Accuracy, Precision, Recall, F1-score and ROC-AUC |
| `gradient_boosting_confusion_matrix.png` | Visualises classification outcomes |
| `gradient_boosting_roc_curve.png` | Visualises ROC performance |

The saved metrics can subsequently be used in the model-comparison stage.

---

## 16. How to Run the Gradient Boosting Notebook

Open:

```text
04_Gradient_Boosting.ipynb
```

Run the notebook cells from top to bottom.

The notebook expects the following input file:

```text
outputs/dormancy_modelling_dataset.csv
```

The required Python libraries include:

- NumPy
- Pandas
- Matplotlib
- scikit-learn

If required, install them with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Then open the notebook in Jupyter Notebook or JupyterLab and execute the cells sequentially.

---

## 17. Reproducibility

The notebook uses the following fixed settings:

```text
Random state = 42
Test size = 0.20
Classification threshold = 0.50
n_estimators = 200
learning_rate = 0.05
max_depth = 3
min_samples_leaf = 20
subsample = 0.80
```

Using the same modelling dataset and configuration should reproduce the model evaluation results.

---

## 18. Summary

The Gradient Boosting model is developed to predict retail investor dormancy using the customer-level modelling dataset produced during feature engineering.

The model uses:

- Median imputation for numerical variables.
- Most-frequent imputation for categorical variables.
- One-hot encoding for categorical variables.
- Balanced sample weights to address class imbalance.
- A Gradient Boosting Classifier with the specified hyperparameters.
- A 0.50 probability threshold for classification.

Model performance is evaluated on a stratified held-out test set using Accuracy, Precision, Recall, F1-score and ROC-AUC, supported by a classification report, confusion matrix and ROC curve.

The complete implementation and performance calculations can be reproduced by running:

`04_Gradient_Boosting.ipynb`
