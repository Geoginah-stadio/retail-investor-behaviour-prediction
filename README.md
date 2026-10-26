# retail-investor-behaviour-prediction
A data science project investigating patterns associated with retail investor dormancy.

## Motivation
The growth of digital investment platforms has made investing more accessible and affordable, supporting greater financial inclusion in South Africa. However, increased registrations do not necessarily translate into consistent investment activity. STADIOEquities faces the challenge of retail investors becoming inactive or dormant, limiting customer retention, business growth and the platform’s financial inclusion objectives. 

This project uses data science to investigate patterns associated with retail investor dormancy by analysing customer characteristics, deposits, transactions and investment activity. The findings may help identify behavioural patterns linked to inactivity and support data-driven strategies to improve investor engagement, retention and long-term participation.

## Statement Problem
Digital investment platforms have made investing more accessible by allowing individuals to open accounts online and invest with relatively small amounts. However, registration does not guarantee sustained investment participation. On STADIOEquities, some users never deposit funds, while others become inactive after initially investing. This gap between registration and continued participation may affect customer retention, revenue generation and the efficient use of marketing resources, while limiting customers' opportunities to build long-term financial security and the platform's contribution to financial inclusion in South Africa (FSCA, 2021; World Economic Forum, 2024; SARB, 2026). 

Although STADIOEquities collects customer and transaction data, including deposits, trading activity, account balances and platform engagement, the behavioural patterns associated with investor dormancy require further investigation. Without understanding these patterns, the platform may struggle to identify investors at risk of becoming inactive and developing effective retention strategies. 

Therefore, this study investigates retail investor dormancy on STADIOEquities using data science and explainable machine learning techniques. It aims to identify behavioural indicators associated with dormancy, evaluate models for identifying investors at risk of inactivity, and explain the key factors influencing sustained investment participation. The findings could support targeted customer engagement, improved retention and data-driven decision-making, contributing to the platform's long-term sustainability and financial inclusion objectives. 

## RAAIDD Log

| RAAIDD | Description |
|----------|-------------|
| Risk | Customer data may contain missing values that could affect model accuracy. |
| Risk | Data may contain duplicate customer records, resulting in biased analysis. |
| Risk | Historical data may not be sufficient to identify long-term engagement patterns. |
| Risk | Customer behaviour may change over time, reducing prediction accuracy. |
| Risk | There may be an imbalance between active and inactive investors in the dataset. |
| Risk | Data privacy regulations may restrict access to certain customer information. |
| Action | Obtain and review all required datasets from STADIOEquities. |
| Action | Perform exploratory data analysis to identify trends and patterns. |
| Action | Cleaning and preparing data for modelling. |
| Action | Creating behavioural features to improve predictive performance. |
| Action | Developing and evaluate multiple machine learning models. |
| Action | Applying explainable AI techniques to interpret model predictions. |
| Action | Present findings and recommendations to stakeholders. |
| Assumption | Customer registration data is accurate and up to date. |
| Assumption | Customer IDs are consistent across all data sources. |
| Assumption | Historical customer behaviour can help predict future behaviour. |
| Assumption | Sufficient historical data is available for analysis. |
| Assumption | Investor activation can be clearly defined and measured. |
| Assumption | Platform engagement activities are recorded consistently. |
| Issue | Delays may occur in receiving the required datasets from the client. |
| Decision | Supervised machine learning techniques such as Logistic Regression and Gradient Boosting will be used to predict investor activation and engagement. |
| Decision | Explainable AI methods will be used to support the interpretation of model predictions. |
| Dependency | A complete data dictionary is required to understand available variables. |
| Dependency | Approval to access and analyse customer data is required before the project can begin. |
| Dependency | Access to customer registration and profile data is required. |
| Dependency | Access to funding and deposit history data is required. |
| Dependency | Access to transaction and trading activity data is required. |


### PROJECT DOCUMENTATION
1. [Preprocessing.MD](https://github.com/Geoginah-stadio/retail-investor-behaviour-prediction/blob/main/notebooks/PreprocessingMD.md)
2. [Feature Engineering.MD](https://github.com/Geoginah-stadio/retail-investor-behaviour-prediction/blob/main/notebooks/FeatureEngineeringMD.MD)
3. [Logistic Regression.MD](https://github.com/Geoginah-stadio/retail-investor-behaviour-prediction/blob/main/models/LogisticRegressionMD.MD)
4. [Gradient Boosting.MD](https://github.com/Geoginah-stadio/retail-investor-behaviour-prediction/blob/main/models/GradientBoostingMD.MD)
5. [Logistic Regression Performance.MD](https://github.com/Geoginah-stadio/retail-investor-behaviour-prediction/blob/main/experimental_results/LogisticRegressionPerformanceMD.MD)
6. [Gradient Boosting Performance.MD](https://github.com/Geoginah-stadio/retail-investor-behaviour-prediction/blob/main/experimental_results/GradientBoostingPerformanceMD.md)
7. [Model Comparison Performance.MD](https://github.com/Geoginah-stadio/retail-investor-behaviour-prediction/blob/main/statistical_analysis/ModelsComparisonMD.MD)

### SS3 Presentation
[CAPTONE PRESENTATION](https://youtu.be/E1QWJRozHqU)
