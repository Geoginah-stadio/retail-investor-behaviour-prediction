# retail-investor-behaviour-prediction
A data science project investigating patterns associated with retail investor activation and dormancy.

## Motivation
The rapid growth of fintech and digital investment platforms has significantly reduced traditional barriers to participating in financial markets. Digital platforms such as STADIOEquities provide individuals with easier and more convenient access to financial products and investment opportunities with relatively small amounts of capital, potentially supporting broader financial inclusion and participation in investment markets (Hong, Lu and Pan, 2025). These developments have created opportunities for more South Africans, particularly among younger and first-time to participate in financial markets and build wealth over time.

Despite these advancements, an important challenge remains, the increasing number of registered accounts does not always lead to meaningful investment participation or long term customer value. STADIOEquities faces a significant business challenge in converting registered users into funded and consistently active investors. The platform has approximately 2.3 million registered accounts, yet a significant proportion of these accounts remain unfunded and inactive after initial participation. This results in a gap between people signing up and actually investing. 

If this issue is not addressed, STADIOEquities may struggle to retain customers and achieve sustainable growth, resulting in lower revenue, efforts to increase financial inclusion may be less effective and investors may also miss opportunities to grow their savings and improve their financial well-being. Understanding and addressing the gap is therefore important for both the commercial sustainability of the platform and its objective of increasing access to investment opportunities (FSCA, 2021). As digital investing becomes more accessible, understanding the factors that encourage or discourage participation becomes critical for ensuring that increased accessibility translates into meaningful financial inclusion rather than merely increased registrations (World Economic Forum, 2024). 

Data science can help address this challenge by analysing information such as account registrations, deposits, transactions, platform usage patterns and customer interactions and providing valuable insights into customer behaviour. By identifying patterns in user behaviour, machine learning models can be used to predict which users are likely to become active investors, fund their accounts and which are at risk of becoming inactive. The use of explainable machine learning is important because it will help STADIOEquities understand the reasons behind these predictions rather than simply generating results.

## Statement Problem
Digital investment platforms have made investing more accessible and convenient by allowing people to open accounts online and start investing with relatively small amounts of money. STADIOEquities has been successful in attracting a large number of registered users, however, many of these users do not progress to becoming active investors. Some users register accounts but never deposit funds, while others who initially invest become inactive over time. This creates a challenge for the platform because high registration numbers do not necessarily translate into meaningful and sustained investment participation (FSCA, 2021; World Economic Forum, 2024). 

This problem affects STADIOEquities and stakeholders, including customers, marketing teams, product managers, and customer support teams. Inactive users may lead to reduced customer retention, lower revenue generation, and inefficient spending on customer acquisition and marketing activities for the organisation. For customers, they represent missed opportunities to participate in investment markets and build long-term financial security. More broadly, low participation among registered users may limit the potential of digital investment platforms to support greater financial inclusion in South Africa (FSCA, 2021; South African Reserve Bank [SARB], 2026). 

STADIOEquities collects a variety of customer data that can be used to investigate this problem. These data include account registration details, funding and deposit records, transaction history, trading activity, account balances and platform engagement information. Analysing these data may reveal patterns and behaviours that identifies users who become active investors and those who remain inactive or disengaged after initial participation. 

Therefore, the aim of this study is to develop and evaluate explainable machine learning techniques that can predict which registered users are likely to become active investors and which active investors are at risk of becoming inactive or dormant on the STADIOEquities platform. In addition, the study aims to identify and explain the key factors influencing investor activation and sustained engagement. The findings could help STADIOEquities develop more targeted strategies to encourage participation, improve customer retention, and support long-term investor engagement through data-driven decision-making.

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
| Action | Clean and prepare data for modelling. |
| Action | Create behavioural features to improve predictive performance. |
| Action | Develop and evaluate multiple machine learning models. |
| Action | Apply explainable AI techniques to interpret model predictions. |
| Action | Present findings and recommendations to stakeholders. |
| Assumption | Customer registration data is accurate and up to date. |
| Assumption | Customer IDs are consistent across all data sources. |
| Assumption | Historical customer behaviour can help predict future behaviour. |
| Assumption | Sufficient historical data is available for analysis. |
| Assumption | Investor activation can be clearly defined and measured. |
| Assumption | Platform engagement activities are recorded consistently. |
| Issue | Delays may occur in receiving the required datasets from the client. |
| Issue | Important variables may contain missing values. |
| Issue | Data formats may differ across systems. |
| Issue | Different departments may define active investors differently. |
| Issue | Available datasets may have limited documentation. |
| Decision | Use supervised machine learning techniques to predict investor activation and engagement. |
| Decision | Use anonymised customer data to ensure privacy compliance. |
| Decision | Compare multiple machine learning algorithms before selecting a final model. |
| Decision | Use behavioural, transactional and engagement variables as predictors. |
| Decision | Use explainable AI methods to support the interpretation of model predictions. |
| Dependency | Access to customer registration and profile data is required. |
| Dependency | Access to funding and deposit history data is required. |
| Dependency | Access to transaction and trading activity data is required. |
| Dependency | A complete data dictionary is required to understand available variables. |
| Dependency | Approval to access and analyse customer data is required before the project can begin. |
