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
| Logistic Regression  | 35.5           | 0.9000         | 0.8985        | 0.6632    | 0.2058  | 0.3141   | 0.7938  |
| Decision Trees       | 28.6           | 0.9023         | 0.9023        | 0.6658    | 0.3847  | 0.3846   | 0.7887  |
| KNN                  | 156.6          | 0.9059         | 0.8979        | 0.6195    | 0.3518  | 0.3518   | 0.7656  |


![image](https://github.com/saikiranbv/Classifiers/blob/main/images/Final_compare_models.png)
![image](https://github.com/saikiranbv/Classifiers/blob/main/images/ROC_compare_models.png)
![image](https://github.com/saikiranbv/Classifiers/blob/main/images/Final_confusion_matrix_models.png)


### Conclusion: Best Model

The Decision Tree model stands out as the best performer for this classification task, especially in terms of the F1 score and test accuracy. This model offers a good balance between precision and recall, making it a strong candidate for deployment. Logistic Regression also performed well and could be considered.


The decision tree model provides insight into different customer segments, guiding targeted marketing strategies:

#### Based on decision tree nodes, when there is low employement, customers are less likely to subscribe
#### When there is more employment and favourable economic conditions  customers are more likely to subscribe
