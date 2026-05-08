# COVID-19 Risk Classification Using PySpark

A big data machine learning project that predicts high-risk COVID-19 scenarios at the province-day level using Apache Spark and PySpark ML. Built as part of the Big Data & AI module at CY Tech (Master 2 ADEO, 2025/2026).

---

## Overview

Early risk identification during epidemic outbreaks is critical for resource allocation and public health intervention. This project frames the problem as binary classification: given demographic, geographical, behavioural, and temporal indicators for a province on a given day, predict whether it represents a high-risk or low-risk COVID-19 scenario.

Five classification algorithms were trained, evaluated, and optimised using cross-validation. Gradient Boosted Trees achieved the best overall performance with 99.19% accuracy and an F1-score of 0.9919.

---

## Dataset

The pipeline integrates multiple heterogeneous data sources joined at the province-day level:

- Province-level COVID-19 confirmed case statistics
- Demographic and geographical information (elderly population ratio, nursing home density)
- Behavioural search trend data (COVID-related query intensity)
- Gender-based confirmed case counts

A percentile-based threshold was applied to confirmed case counts to generate the binary target variable (high-risk / low-risk).

---

## Pipeline

```
Raw Data Sources
     ↓
Data Preprocessing (missing values, encoding, scaling, leakage prevention)
     ↓
Feature Engineering & Selection (domain-guided, validated with Random Forest importances)
     ↓
Model Training (5 classifiers, baseline + cross-validated)
     ↓
Evaluation (Accuracy, F1-score, AUROC, Avg Precision-Recall)
     ↓
Risk Classification Output (High / Low)
```

---

## Models

Five classifiers were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosted Trees
- Neural Network (Multilayer Perceptron)

Hyperparameter tuning was done via grid search with cross-validation.

---

## Results

### Baseline

| Model | Accuracy | F1-score |
|---|---|---|
| Logistic Regression | 0.9654 | 0.9654 |
| Decision Tree | 0.8380 | 0.8371 |
| Random Forest | 0.8202 | 0.8171 |
| Gradient Boosted Trees | **0.9914** | **0.9914** |
| Neural Network | 0.7115 | 0.6918 |

### After Cross-Validation Optimisation

| Model | Accuracy | F1-score |
|---|---|---|
| Logistic Regression | 0.9501 | 0.9501 |
| Decision Tree | 0.9910 | 0.9910 |
| Random Forest | 0.9059 | 0.9057 |
| Gradient Boosted Trees | **0.9919** | **0.9919** |
| Neural Network | 0.8098 | 0.8081 |

Gradient Boosted Trees was selected as the final model based on its accuracy, F1-score, AUROC, and average precision-recall performance under class imbalance.

---

## Key Features

EDA and Random Forest-based importance analysis identified the most influential predictors:

- Elderly population ratio
- Nursing home count/density
- COVID-related online search trend intensity

These are consistent with known epidemiological risk factors — regions with higher elderly populations and denser institutional care are more vulnerable, and rising search activity tends to correlate with public awareness during outbreak surges.

---

## Tech Stack

- Python
- Apache Spark / PySpark
- PySpark ML (pipelines, cross-validation, classifiers)
- Pandas, Matplotlib (EDA and visualisation)
- Jupyter Notebook

---

## Deployment Architectures Considered

The pipeline is designed to be scalable across multiple deployment environments:

- **Databricks** — batch inference with cloud object storage
- **AWS EMR + S3** — flexible cluster configuration
- **Azure Synapse Analytics** — integrated Spark pools with cloud analytics
- **On-Premise Spark Cluster** — for strict data governance requirements.

---

## Author

Harsh Prajapati — [github.com/harshbaee](https://github.com/harshbaee)  
Master 2 ADEO, CY Tech, Cergy
