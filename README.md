# 🚀 Income Classification & Analysis Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![R-to-Python](https://img.shields.io/badge/Refactor-R_to_Python-green)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)

## 📖 Overview
This project is a comprehensive refactor of a statistical analysis study (originally developed in R for STAT 412) into a production-ready **Python Machine Learning Pipeline**.

The objective is to predict whether an individual's annual income exceeds $50K based on census data (`salary.csv`), utilizing advanced feature engineering and supervised learning algorithms.

## 🔄 Project Evolution (R → Python)
The original study utilized **R (dplyr, caret, VIM)** for statistical inference. This repository modernizes the workflow using **Python (Pandas, Scikit-Learn Pipelines)** to demonstrate:
- **Reproducibility:** End-to-end script from raw data to model evaluation.
- **Data Engineering:** Handling missing values (`?` to `NaN`), grouping sparse categories (e.g., Government jobs), and KNN imputation.
- **Benchmarking:** Comparing Logistic Regression, Decision Trees, and Random Forest.

## 🛠️ Key Features
* **Preprocessing Pipeline:** * Automatic handling of missing values (imputation).
    * Grouping high-cardinality features (e.g., `workclass`, `marital-status`).
    * One-Hot Encoding for categorical variables.
* **Model Evaluation:** Precision, Recall, F1-Score, and Confusion Matrix analysis.

## 📊 Results
Based on the test set evaluation, the **Random Forest Classifier** yielded the best balance between precision and recall.

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| **Random Forest** | **~85%** | **0.84** |
| Logistic Regression | ~83% | 0.82 |
| Decision Tree | ~81% | 0.80 |

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mrkn7/income-prediction-pipeline.git](https://github.com/mrkn7/income-prediction-pipeline.git)
