# 📊 Benchmarking Supervised Learning for Income Classification

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 Overview
This project implements an end-to-end Machine Learning pipeline to predict whether an individual's annual income exceeds $50K based on census data. The study focuses on benchmarking various supervised learning algorithms and interpreting model decisions using feature importance analysis.

Unlike standard implementations, this project utilizes **Scikit-Learn Pipelines** for robust preprocessing and prevents data leakage.

## 🎯 Objectives
- **Preprocessing:** Handling missing values, encoding categorical variables, and scaling numerical features using `ColumnTransformer`.
- **Modeling:** Comparing Logistic Regression, Decision Trees, and Random Forest Classifiers.
- **Evaluation:** Assessing performance using Accuracy, F1-Score, and ROC-AUC metrics.

## 🛠️ Technologies Used
- **Python:** Core programming language.
- **Pandas & NumPy:** Data manipulation.
- **Scikit-Learn:** Modeling and preprocessing pipelines.
- **Matplotlib & Seaborn:** Exploratory Data Analysis (EDA).

## 📊 Model Performance
The Random Forest Classifier achieved the highest performance metrics, demonstrating robustness against overfitting compared to simpler models.

| Model | Accuracy | F1-Score (Weighted) | ROC-AUC |
|-------|----------|---------------------|---------|
| **Random Forest** | **85.3%** | **0.84** | **0.90** |
| Logistic Regression | 84.1% | 0.83 | 0.88 |
| Decision Tree | 81.5% | 0.81 | 0.74 |

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mrkn7/income-prediction-benchmark.git](https://github.com/mrkn7/income-prediction-benchmark.git)
