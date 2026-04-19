# Diabetes Prediction — ML Classification Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange) ![Status](https://img.shields.io/badge/Status-Complete-green)

Predicting diabetes diagnosis from patient health records by comparing three machine learning classification models. Built on a dataset of 100,000 patients with features including blood glucose levels, HbA1c, BMI, age, and medical history.

## Results

| Model | Accuracy | Precision (diabetes) | Recall (diabetes) | F1 (diabetes) |
|---|---|---|---|---|
| **Random Forest** | **97%** | **93%** | **67%** | **0.78** |
| Logistic Regression | 95% | 80% | 62% | 0.70 |
| K-Nearest Neighbors | 95% | 87% | 53% | 0.66 |

**Winner: Random Forest Classifier.** While all three models achieved similar overall accuracy (~95–97%), Random Forest had the highest recall on positive diabetes cases. In medical diagnosis, recall is the critical metric — a false negative (telling a diabetic patient they don't have diabetes) is far more harmful than a false positive.

## Key findings

- **HbA1c level** and **blood glucose level** were the strongest predictors of diabetes (highest OLS regression coefficients: 0.0833 and 0.0023 respectively, both p < 0.001)
- **Heart disease** and **hypertension** showed statistically significant positive associations with diabetes (coefficients: 0.113 and 0.092)
- **Age** and **BMI** were also significant predictors (p < 0.001), consistent with clinical literature
- **Class imbalance:** ~91.5% of patients are non-diabetic vs ~8.5% diabetic — a limitation that suppresses recall scores across all models
- No multicollinearity detected across predictors via correlation heatmap

## Project structure

```
diabetes-prediction-ml/
├── diabetes_prediction_analysis.ipynb   # Main notebook
└── README.md
```

## Notebook walkthrough

- **Part 1–3 — Setup & objectives:** Defined exploratory and predictive objectives; loaded 100,000-row Kaggle dataset
- **Part 4 — Data preprocessing:** Removed 3,854 duplicates; encoded gender (Female→0, Male→1); converted smoking_history to ordinal scale (never/No Info→0, former/ever/not current→1, current→2)
- **Part 5 — EDA:** Class distribution countplot, age histogram, BMI vs blood glucose scatter plot, correlation heatmap
- **Part 6 — Modeling:** Logistic Regression, KNN (k=5), Random Forest — all evaluated with confusion matrix and classification report
- **Part 7–8 — Validation & findings:** Model comparison, OLS regression for statistical significance, coefficient analysis

## Dataset

- **Source:** Kaggle — Diabetes Prediction Dataset
- **Size:** 100,000 observations → 96,128 after deduplication
- **Features:** gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level
- **Target:** diabetes (binary: 0 = No, 1 = Yes)

## Tech stack

Python · Pandas · NumPy · Scikit-learn · Seaborn · Matplotlib · Statsmodels · Google Colab

## How to run

1. Open `diabetes_prediction_analysis.ipynb` in Google Colab
2. Download the dataset from Kaggle and upload to your Google Drive
3. Update the file path in cell 2: `df = pd.read_csv("/content/drive/MyDrive/diabetes_prediction_dataset.csv")`
4. Runtime → Run all

## Potential improvements

- Apply SMOTE oversampling to address class imbalance and improve recall on diabetic cases
- Add SHAP values to explain which features drive individual predictions
- Hyperparameter tuning via GridSearchCV on Random Forest (n_estimators, max_depth)
- Try XGBoost or LightGBM for comparison
