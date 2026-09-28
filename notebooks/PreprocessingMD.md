# Data Preprocessing

## 1. Overview

This document describes the data preprocessing performed for the
research project:

**A Machine Learning Approach to Predicting Retail Investor Dormancy on
Digital Investment Platforms**

The purpose of the preprocessing stage is to prepare the raw customer
and transaction datasets for the later feature-engineering and
machine-learning modelling stages.

The preprocessing stage focuses on:

-   Loading the customer and transaction datasets.
-   Inspecting the structure and data quality of the datasets.
-   Converting transaction dates to the correct format.
-   Removing records that cannot be used for temporal analysis.
-   Checking for duplicate records.
-   Checking for missing values.
-   Inspecting transaction variables for possible impossible or missing
    values.
-   Separating transaction data into observation and prediction periods.
-   Saving clean datasets for the feature-engineering stage.

**Important:** The `dormant` target variable is not created during this
preprocessing stage. It is created in the feature-engineering stage
because it depends on customer-level transaction behaviour across two
time periods.

------------------------------------------------------------------------

## 2. Preprocessing Notebook

The preprocessing code is contained in:

`01_Data_Preprocessing.ipynb`

The notebook uses Python and the following main libraries:

-   `pandas`
-   `numpy`
-   `pathlib`

The notebook was developed using Python 3.12.5.

------------------------------------------------------------------------

## 3. Input Data

Two raw datasets are used:

### 3.1 Transaction Data

File:

`transactions_data.csv`

The transaction dataset contains four variables:

-   `customer_id`
-   `date`
-   `amount`
-   `type`

The raw dataset contains **3,159,157 transaction records**.

### 3.2 Customer Data

File:

`customer_data.csv`

The customer dataset contains **48,723 customers** and **54 variables**.

The variables include customer demographic information, product
information, transaction-related information, satisfaction measures,
customer behaviour and other customer-level attributes.

Both datasets are loaded using Pandas.

------------------------------------------------------------------------

## 4. Data Inspection

The first stage of preprocessing is to inspect the structure and quality
of the raw datasets.

The notebook:

-   Displays the first records of both datasets.
-   Checks the number of rows and columns.
-   Displays the column names.
-   Checks data types.
-   Checks the number of non-null observations.
-   Checks for missing values.
-   Checks for duplicate records.
-   Checks whether `customer_id` is duplicated in the customer dataset.
-   Produces descriptive statistics for the transaction variables.

This inspection is used to identify data-quality issues before the data
is used for modelling.

------------------------------------------------------------------------

## 5. Missing Values

Missing values are checked separately for the transaction and customer
datasets.

### Transaction data

The transaction dataset contains no missing values in:

-   `customer_id`
-   `date`
-   `amount`
-   `type`

### Customer data

Missing values are present in three customer variables:

  Variable                       Missing values
  ---------------------------- ----------------
  `complaint_topics`                     24,346
  `credit_utilization_ratio`             18,263
  `feature_requests`                     16,115

The notebook identifies these missing values during the data-quality
inspection stage.

No missing-value imputation is performed in this preprocessing notebook.

------------------------------------------------------------------------

## 6. Duplicate Records

The notebook checks for duplicate records in both datasets.

The results were:

-   **102 duplicate transaction rows**
-   **0 duplicate customer rows**
-   **0 duplicate customer IDs**

The 102 exact duplicate transaction records are removed before the
transaction data is used for the temporal analysis.

------------------------------------------------------------------------

## 7. Date Conversion

The transaction `date` variable is converted from its original format to
Pandas datetime format.

The conversion uses:

``` python
pd.to_datetime(transactions["date"], errors="coerce")
```

The notebook then checks whether any dates could not be converted.

The result was:

**Invalid/missing transaction dates: 0**

Records with missing `customer_id` or `date` are removed because these
fields are required for the temporal analysis.

------------------------------------------------------------------------

## 8. Removing Duplicate Transactions

After checking the transaction data, exact duplicate transaction records
are removed using:

``` python
transactions = transactions.drop_duplicates().copy()
```

After this step, the clean transaction dataset contains:

**3,159,055 transaction records**

------------------------------------------------------------------------

## 9. Transaction Data Inspection

The notebook uses descriptive statistics to inspect the transaction
variables:

``` python
transactions.describe(include="all").T
```

This provides information such as:

-   number of observations,
-   unique values,
-   most frequent values,
-   mean,
-   minimum,
-   maximum,
-   standard deviation, and
-   quartiles where applicable.

The purpose is to inspect the transaction variables for potential
data-quality issues before continuing with the analysis.

------------------------------------------------------------------------

## 10. Observation and Prediction Periods

A key part of the preprocessing is separating historical transaction
activity from future transaction activity.

The notebook uses:

``` text
Cut-off date: 30 September 2023
Prediction end: 29 December 2023
```

### Observation period

The observation period covers:

**4 January 2023 to 30 September 2023**

There are:

**2,367,794 observation-period transactions**

These transactions represent the information available before the
prediction point.

### Prediction period

The prediction period covers:

**1 October 2023 to 29 December 2023**

There are:

**791,261 prediction-period transactions**

These later transactions are used in the feature-engineering stage to
establish the future dormancy outcome.

This separation ensures that future transaction information is not used
as a predictor during the observation period.

------------------------------------------------------------------------

## 11. Output Files

After preprocessing, three clean datasets are saved in the `outputs/`
directory.

### `observation_transactions.csv`

Contains transaction activity available before the prediction point.

### `prediction_transactions.csv`

Contains later transaction activity that is used to establish the future
dormancy outcome.

### `clean_customer_data.csv`

Contains the cleaned customer-level information.

These files are used by the next stage of the project for feature
engineering and creation of the binary `dormant` target.

------------------------------------------------------------------------

## 12. Feature Engineering and Target Variable

Feature engineering is performed after this preprocessing stage.

The preprocessing notebook does **not** create the final customer-level
behavioural features or the `dormant` target.

The next stage uses the three output files to:

-   create customer-level behavioural features, and
-   create the binary `dormant` target based on transaction behaviour
    across the relevant time periods.

This separation keeps data cleaning and preparation separate from
feature engineering and target creation.

------------------------------------------------------------------------

## 13. How to Run the Preprocessing Notebook

### Requirements

The notebook requires:

-   Python 3.12.5
-   Jupyter Notebook or JupyterLab
-   Pandas
-   NumPy

### Step 1: Place the files in the project directory

The project directory should contain:

``` text
01_Data_Preprocessing.ipynb
transactions_data.csv
customer_data.csv
```

The notebook creates the `outputs/` directory automatically.

### Step 2: Open the notebook

Open:

`01_Data_Preprocessing.ipynb`

using Jupyter Notebook, JupyterLab, or another compatible notebook
environment.

### Step 3: Run the notebook

Run the notebook cells from beginning to end.

The notebook will:

1.  Load the raw datasets.
2.  Inspect the datasets.
3.  Check missing values.
4.  Check duplicate records.
5.  Convert transaction dates.
6.  Remove unusable and duplicate transaction records.
7.  Inspect transaction statistics.
8.  Separate observation and prediction periods.
9.  Save the three clean output datasets.

------------------------------------------------------------------------

## 14. Preprocessing Outcome

The preprocessing stage produces three clean files:

-   `observation_transactions.csv`
-   `prediction_transactions.csv`
-   `clean_customer_data.csv`

These files provide the prepared data required for the
feature-engineering and modelling stages of the retail investor dormancy
prediction project.
