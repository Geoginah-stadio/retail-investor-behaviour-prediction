# Feature Engineering

## 1. Overview

**Research topic:** *A Machine Learning Approach to Predicting Retail Investor Dormancy on Digital Investment Platforms*

This document describes the feature-engineering process used to transform the preprocessed transaction and customer data into a customer-level modelling dataset.

The feature-engineering calculations are contained in:

`02_Feature_Engineering(1).ipynb`

The final modelling dataset produced by the notebook is:

`outputs/dormancy_modelling_dataset.csv`

The resulting dataset contains one row per investor and is used as the input to the machine-learning models.

---

## 2. Feature Engineering Notebook

The complete feature-engineering implementation is contained in:

`02_Feature_Engineering(1).ipynb`

The notebook performs the following main steps:

1. Loads the observation-period transactions.
2. Loads the prediction-period transactions.
3. Loads the cleaned customer data.
4. Creates customer-level transaction features.
5. Creates transaction-type count features.
6. Creates behavioural features.
7. Creates the dormancy target using future transaction activity.
8. Merges customer-level characteristics with transaction features.
9. Standardises selected monetary variables.
10. Checks for missing values.
11. Checks the final target distribution.
12. Saves the final modelling dataset.

---

## 3. Observation and Prediction Periods

The notebook uses two transaction datasets:

- `observation_transactions.csv`
- `prediction_transactions.csv`

The observation-period transactions are used to create the model predictors.

The prediction-period transactions are **not used to create predictor variables**. They are used only to determine whether an investor subsequently became dormant.

The notebook defines the observation cut-off date as:

```python
CUTOFF_DATE = pd.Timestamp("2023-09-30")
```

This separation is important because the features must represent information available before the prediction period.

---

## 4. Definition of the Target Variable

The target variable is:

`dormant`

It is binary:

| Value | Meaning |
|---:|---|
| `0` | Investor continued to transact during the prediction period |
| `1` | Investor had transaction activity during the observation period but no transaction during the prediction period |

The target is created by identifying customers present in the prediction-period transaction data.

The notebook implements this as:

```python
future_customers = set(prediction["customer_id"].unique())

features["dormant"] = (
    ~features["customer_id"].isin(future_customers)
).astype(int)
```

This means the prediction-period data is used only for target creation and not as a predictor.

---

## 5. Customer-Level Transaction Features

The observation transactions are aggregated by `customer_id`.

The following historical transaction features are created:

| Feature | Description |
|---|---|
| `pre_tx_count` | Number of transactions during the observation period |
| `pre_total_volume` | Total transaction amount during the observation period |
| `pre_avg_tx_value` | Average transaction value |
| `pre_median_tx_value` | Median transaction value |
| `pre_std_tx_value` | Standard deviation of transaction values |
| `pre_max_tx_value` | Maximum transaction value |
| `pre_active_days` | Number of distinct days on which the investor transacted |
| `pre_first_tx` | Date of the investor's first observation-period transaction |
| `pre_last_tx` | Date of the investor's most recent observation-period transaction |

The aggregation is performed using the transaction `amount` and `date` fields.

For customers with only one transaction, the standard deviation is undefined. These missing standard-deviation values are replaced with zero:

```python
features["pre_std_tx_value"] = (
    features["pre_std_tx_value"].fillna(0)
)
```

---

## 6. Transaction-Type Features

Transaction types are counted for each customer using a cross-tabulation of:

- `customer_id`
- `type`

The resulting features use the following naming convention:

```text
pre_<transaction_type>_count
```

The notebook explicitly ensures the following expected features exist:

| Feature | Description |
|---|---|
| `pre_deposit_count` | Number of deposits |
| `pre_withdrawal_count` | Number of withdrawals |
| `pre_payment_count` | Number of payments |
| `pre_transfer_count` | Number of transfers |

If an expected transaction type does not occur in the data, its feature is created and assigned a value of zero.

---

## 7. Behavioural Features

The notebook derives additional behavioural features from the historical transaction data.

### 7.1 Recency

`pre_recency_days`

Measures the number of days between the observation cut-off date and the investor's most recent observation-period transaction.

```python
pre_recency_days = CUTOFF_DATE - pre_last_tx
```

A larger value indicates that the investor's most recent transaction occurred further before the cut-off date.

---

### 7.2 Transaction Intensity

`pre_tx_per_active_day`

Measures the average number of transactions per active transaction day.

```text
Transaction intensity =
Number of transactions / Number of active days
```

The calculation uses `pre_active_days` and protects against division by zero.

---

### 7.3 Transaction-Type Ratios

The notebook converts transaction-type counts into proportions of total transactions.

The following features are created:

- `pre_deposit_ratio`
- `pre_withdrawal_ratio`
- `pre_payment_ratio`
- `pre_transfer_ratio`

For example:

```text
pre_deposit_ratio =
pre_deposit_count / pre_tx_count
```

These ratios provide behavioural information about the composition of an investor's transaction activity.

---

## 8. Customer Characteristics

The transaction-derived features are merged with available customer-level characteristics from:

`clean_customer_data.csv`

The notebook attempts to include the following fields:

| Customer characteristic |
|---|
| `customer_id` |
| `age` |
| `gender` |
| `location` |
| `income_bracket` |
| `occupation` |
| `education_level` |
| `marital_status` |
| `household_size` |
| `acquisition_channel` |
| `savings_account` |
| `credit_card` |
| `personal_loan` |
| `investment_account` |
| `insurance_product` |

Only columns that are actually available in the customer dataset are selected.

The merge is performed at customer level using `customer_id`.

---

## 9. Standardisation of Monetary Variables

Selected monetary variables are standardised using scikit-learn's `StandardScaler`.

The variables selected for standardisation are:

- `pre_total_volume`
- `pre_avg_tx_value`
- `pre_median_tx_value`
- `pre_std_tx_value`
- `pre_max_tx_value`

For each selected variable, a corresponding standardised feature is created using the prefix:

`std_`

For example:

```text
pre_total_volume
        ↓
std_pre_total_volume
```

The standardisation is performed using:

```python
scaler = StandardScaler()
```

The purpose is to place the selected monetary variables on a comparable scale, which is particularly relevant for models such as Logistic Regression.

---

## 10. Date Variables

The raw transaction date fields:

- `pre_first_tx`
- `pre_last_tx`

are used during feature engineering but are removed from the final modelling dataset.

The information contained in `pre_last_tx` that is relevant for modelling is retained through:

`pre_recency_days`

This converts the transaction date into a numerical behavioural measure.

---

## 11. Missing-Value Check

After feature engineering and merging, the notebook checks the modelling dataset for missing values.

The check is performed using:

```python
missing = (
    modelling_data.isna()
    .sum()
    .sort_values(ascending=False)
)
```

Only columns with missing values are displayed.

The notebook itself does not apply a general missing-value imputation step at this stage. Missing values that remain are therefore identified for subsequent handling by the modelling pipeline.

---

## 12. Target Distribution Check

The notebook calculates the final distribution of the `dormant` target.

Both counts and percentages are displayed for:

- Non-dormant (`0`)
- Dormant (`1`)

This provides an initial check of class balance before model training.

The exact counts and percentages depend on the supplied input datasets and are generated when the notebook is executed.

---

## 13. Final Modelling Dataset

The completed feature-engineering process produces:

`outputs/dormancy_modelling_dataset.csv`

The dataset contains:

- Historical transaction behaviour
- Transaction frequency measures
- Monetary transaction measures
- Transaction-type counts
- Transaction-type ratios
- Recency
- Transaction intensity
- Customer characteristics
- Standardised monetary variables
- The binary `dormant` target

The final dataset is structured at **one row per investor**.

This dataset is then used as the input for the Logistic Regression and Gradient Boosting modelling stages.

---

## 14. Output

The notebook saves the final dataset using:

```python
output_file = OUTPUT_DIR / "dormancy_modelling_dataset.csv"

modelling_data.to_csv(output_file, index=False)
```

The output location is:

```text
outputs/dormancy_modelling_dataset.csv
```

The notebook also prints the final number of rows and columns after feature engineering.

---

## 15. How to Run the Feature Engineering

Open:

`02_Feature_Engineering(1).ipynb`

Run the notebook cells from top to bottom.

The notebook expects the following input files inside the `outputs/` directory:

```text
outputs/
├── observation_transactions.csv
├── prediction_transactions.csv
└── clean_customer_data.csv
```

After successful execution, the following modelling dataset is created:

```text
outputs/
└── dormancy_modelling_dataset.csv
```

### Required Python Libraries

The notebook uses:

- NumPy
- Pandas
- scikit-learn

If required, install them with:

```bash
pip install numpy pandas scikit-learn
```

Then open the notebook in Jupyter Notebook or JupyterLab and execute the cells sequentially.

---

## 16. Feature Engineering Summary

The feature-engineering process converts transaction-level and customer-level information into customer-level behavioural predictors.

The key feature groups are:

| Feature group | Examples |
|---|---|
| Transaction volume | `pre_tx_count`, `pre_total_volume` |
| Transaction value | `pre_avg_tx_value`, `pre_median_tx_value`, `pre_max_tx_value` |
| Transaction variability | `pre_std_tx_value` |
| Activity | `pre_active_days`, `pre_tx_per_active_day` |
| Recency | `pre_recency_days` |
| Transaction type | Deposit, withdrawal, payment and transfer counts |
| Transaction mix | Deposit, withdrawal, payment and transfer ratios |
| Customer profile | Age, gender, income bracket, occupation, education, etc. |
| Product ownership | Savings account, credit card, loan, investment and insurance indicators |
| Standardised monetary variables | `std_pre_*` features |
| Target | `dormant` |

---

## 17. Reproducibility

The feature-engineering process can be reproduced by running:

`02_Feature_Engineering(1).ipynb`

using the required input files in the expected `outputs/` directory.

The notebook defines the observation cut-off date as:

```text
2023-09-30
```

The resulting dataset should be generated at:

```text
outputs/dormancy_modelling_dataset.csv
```

No feature values or target-distribution statistics are manually entered in this document; they are generated from the underlying input datasets when the notebook is executed.
