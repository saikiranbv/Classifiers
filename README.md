# Bank Marketing Analysis with Classifiers

## Overview

This practical application assignment aims to compare the performance of various classification algorithms to predict whether a client will subscribe to a long-term deposit based on data from multiple marketing campaigns conducted by a Portuguese banking institution. The classifiers compared in this project include Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees, and Support Vector Machines.

Here is a link to the Jupyter Notebook https://github.com/saikiranbv/Classifiers/blob/main/Bank_Marketing_analysis.ipynb

## Table of Contents

1. [Business Understanding](#business-understanding)
2. [Data Understanding](#data-understanding)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
5. [Evaluation](#evaluation)
6. [Deployment](#deployment)
7. [Usage](#usage)
8. [Repository Structure](#repository-structure)
9. [Notebook](#notebook)

## Business Understanding

The primary objective is to optimize the direct marketing efforts of the bank by accurately predicting whether a client will subscribe to a long-term deposit. By achieving this, the bank can improve the efficiency of its marketing campaigns, reduce costs, and enhance customer targeting, leading to higher profitability and better resource allocation.

## Data Analysis

This is from UCI Machine Learning Repository https://archive.ics.uci.edu/dataset/222/bank+marketing. The dataset 'bank-additional-full.csv' comprises data collected from 17 marketing campaigns conducted between May 2008 and November 2010, resulting in 41,188 records and 20 inputs.

### Potential Issues:

- `unknown` values in categorical variables such as `job`, `education`, `default`, `housing`, and `loan` represent missing data.
- `999` in `pdays` indicates that the client was not previously contacted.

## Data Preparation


1. **Data Cleaning:**
   - Removed 12 duplicate rows from the dataset.

2. **Correlation Analysis:**
    - A correlation matrix was calculated for the numerical features to understand the relationships between variables. Some key insights include:
      - `pdays` and `previous` show a strong negative correlation (-0.59).
      - `emp.var.rate`, `euribor3m`, and `nr.employed` are strongly positively correlated with each other.
      - `duration` has a minimal correlation with other features but is crucial in predicting the target variable.

3. **Splitting Data:**
    - The dataset was split into training and test sets to evaluate model performance.

4. **Feature Engineering: Preprocessing Pipeline**
    - Numerical features were scaled using `StandardScaler`.
    - Categorical features were encoded using `OrdinalEncoder`.

## Model Training

### Models Compared:
  - **K-Nearest Neighbors (KNN)**
  - **Logistic Regression**
  - **Decision Trees**
  - **Support Vector Machines (SVM)**

### Initial Model Comparisons


| Model                  | Train Time | Train Accuracy | Test Accuracy | Precision | Recall  | F1-Score | AUC    |
|------------------------|------------|----------------|---------------|-----------|---------|----------|--------|
| Logistic Regression    | 0.16       | 0.90           | 0.90          | 0.66      | 0.20    | 0.31     | 0.79   |
| Decision Tree          | 0.12       | 0.99           | 0.84          | 0.30      | 0.33    | 0.33     | 0.63   |
| KNN                    | 0.006      | 0.91           | 0.89          | 0.55      | 0.29    | 0.36     | 0.72   |
| SVM                    | 192.78     | 0.90           | 0.90          | 0.67      | 0.18    | 0.29     | 0.71   |

![image](https://github.com/saikiranbv/Classifiers/blob/main/images/Init_compare_models.png)
![image](https://github.com/saikiranbv/Classifiers/blob/main/images/Init_confusion_matrix_models.png)

  - **Logistic Regression** had the good performance and **Decision Tree** was similar
  - **KNN** was fastest but had lower F1 and AUC.
  - **SVM** computationally very expensive as compared to other models and had lowest F1.

### Model Optimization
Hyperparameter tuning was performed using cross-validation and Grid Search:
  - **Logistic Regression**: Best parameters found were 'C': 10, 'penalty': 'l2', 'solver': 'liblinear'
  - **Decision Tree**: Best parameters were 'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 10
  - **KNN**: Optimal parameters were 'n_neighbors': 13, 'p': 1, 'weights': 'uniform'
  - **SVM**: The  model was computationally expensive for the given dataset, not evaluated further due to compute constraints.

## Evaluation

### Metrics Used:
  - **Accuracy**: The proportion of correctly classified instances.
  - **Precision, Recall, F1-Score**: To assess the balance between false positives and false negatives.
  - **ROC-AUC**: To evaluate the model's ability to distinguish between the two classes.

### Model Performance Summary
The following table summarizes the performance metrics for the improved models:

| Model                | Train Time (s) | Train Accuracy | Test Accuracy | Precision | Recall  | F1 Score | AUC     |
|----------------------|----------------|----------------|---------------|-----------|---------|----------|---------|
| Logistic Regression  | 39.04          | 0.9107         | 0.9098        | 0.6576    | 0.4159  | 0.5096   | 0.9333  |
| Decision Trees       | 52.84          | 0.9175         | 0.9142        | 0.6503    | 0.5151  | 0.5749   | 0.9303  |
| KNN                  | 541.80         | 1.0000         | 0.9024        | 0.6330    | 0.3179  | 0.4232   | 0.9129  |



![image](https://github.com/user-attachments/assets/189d8944-f234-4f19-9b1a-f64bfad21e71)
![image](https://github.com/user-attachments/assets/054f1247-d888-4bd3-853e-c45b2856b461)
![image](https://github.com/user-attachments/assets/8054c8e5-ae7f-427b-8ba1-ba5a1604362c)

### Key Findings

1. **Logistic Regression**:
   - Achieved a balanced performance with a strong AUC of 0.9333, indicating good model discrimination between classes.
   - Precision and recall are moderate, with room for improvement in the recall to capture more true positives.
     
2. **Decision Trees**:
   - The Decision Tree model delivered the best F1 Score of 0.5749, balancing precision and recall effectively.
   - This model also showed robust performance across all metrics, with the highest test accuracy of 0.9142 and AUC 0.9303.

3. **K-Nearest Neighbors (KNN)**:
   - While KNN achieved perfect train accuracy, indicating it might be overfitting, its test accuracy was slightly lower at 0.9024.
   - KNN has the lowest recall at 0.3179, indicating it missed many true positives, but it maintained a decent AUC.

### Conclusion: Best Model

The Decision Tree model stands out as the best performer for this classification task, especially in terms of the F1 score and test accuracy. This model offers a good balance between precision and recall, making it a strong candidate for deployment. Logistic Regression also performed well and could be considered depending on the specific use case requirements. However, KNN, despite its high training accuracy, may require further tuning or regularization to improve generalization and recall.

These findings suggest that feature engineering, further hyperparameter tuning, and model selection will be crucial in optimizing performance for this dataset.

## Decision Tree Interpretation
![image](https://github.com/user-attachments/assets/21b1c9c3-a645-4c43-86c5-3dd420d14730)
![image](https://github.com/user-attachments/assets/dffc17a3-778d-4717-ac6f-2771702976dc)

The decision tree model provides insight into different customer segments, guiding targeted marketing strategies:

#### Segment 1: Low Employment, Short Call Duration
- **Characteristics**: Customers in regions with low employment rates and short call durations.
- **Behavior**: Less likely to subscribe unless call duration is slightly longer.
- **Recommendations**: Engage customers by extending conversations and focusing on understanding their needs.

#### Segment 2: Low Employment, Long Call Duration
- **Characteristics**: Customers in low employment regions with longer call durations.
- **Behavior**: More likely to subscribe, especially if there have been few prior contacts.
- **Recommendations**: Use the extended call time to build a strong connection and present term deposits as a secure investment option.

#### Segment 3: High Employment, Short Call Duration
- **Characteristics**: Customers in high employment regions but with short call durations.
- **Behavior**: Less likely to subscribe, especially with low economic confidence.
- **Recommendations**: Address concerns early in the conversation and suggest follow-up emails or digital resources.

#### Segment 4: High Employment, Long Call Duration with Favorable Economic Conditions
- **Characteristics**: Customers in regions with high employment, long call durations, and favorable economic conditions.
- **Behavior**: More likely to subscribe, particularly if economic indicators are positive.
- **Recommendations**: Emphasize the benefits of term deposits and introduce related financial products.

#### Segment 5: High Employment, Long Call Duration with Unfavorable Economic Conditions

- **Characteristics**: Customers in regions with high employment, long call durations, but facing less favorable economic indicators.
- **Behavior**: Decision to subscribe is nuanced and influenced by economic conditions.
- **Recommendations**: Provide products with flexibility and stress the security of the investment.

## Marketing Strategies

  - **Tailored Campaigns:** Develop targeted marketing campaigns based on customer segments, focusing on their specific needs and preferences.
  - **Call Duration Optimization:** Implement strategies to increase call duration for segments with lower conversion rates.
  - **Economic Indicators:** Continuously monitor economic indicators and adjust marketing strategies accordingly.
  - **Agent Training:** Equip sales agents with knowledge about different customer segments and how to tailor their approach.
  - **Customer Relationship Management (CRM):** Utilize CRM systems to track customer interactions, preferences, and purchase history for personalized marketing.

## Deployment

### Model Deployment:

- The final model is ready for deployment, allowing the bank to predict the likelihood of clients subscribing to long-term deposits. The model can be integrated into the bank's CRM system for real-time decision-making.

## Usage

### Requirements:

- Python 3.x
- pandas, numpy, scikit-learn, matplotlib, seaborn (Python libraries)

### Running the Project:

**Clone the Repository:**
   ```bash
   git clone https://github.com/mitbans/Bank-Marketing-Campaigns-Analysis.git
   ```

### Further Steps:
- Deploy the model using a web application framework like Flask or Streamlit for user-friendly interaction.

## Repository Structure
- <code>data/bank-additional-full.csv</code>: Contains dataset used in the analysis.
- <code>images/decision_tree.pdf</code>: Contains final Decision Tree Model.
- <code>notebooks/Predicting-Long-Term-Deposit-Success.ipynb</code>: Jupyter notebook with code for data analysis.
- <code>README.md</code>: Summary of findings and link to notebook

## Notebook
The detailed analysis and code can be found in the Jupyter notebook <a href="https://github.com/mitbans/Bank-Marketing-Campaigns/blob/main/notebooks/Predicting-Long-Term-Deposit-Success.ipynb">here</a>.
