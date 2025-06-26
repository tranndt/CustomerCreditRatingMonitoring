# 🧠 Credit Risk Classification – ML Modeling & Feature Engineering

This folder contains the machine learning pipeline used to classify accounts into risk groups (A, B, C) based on their historical behavior.

## 🔍 Problem

How can we classify customer accounts into actionable risk groups — before they default?  
Our goal: Predict whether an account is low-risk, moderate, or likely to default using only their historical behavior.

## 🧪 Key Components

| Notebook | Purpose |
|----------|---------|
| `01_preview_data.ipynb` | Explore and understand the raw data |
| `02_transform_data.ipynb` | Engineer features from balance snapshots |
| `03_label_accounts.ipynb` | Apply hybrid logic-based labeling: A (good), B (neutral), C (risky) |
| `04_train_model.ipynb` | Train Random Forest classifier on behavioral features |
| `05_results_analysis.ipynb` | Evaluate accuracy and interpretability (SHAP, LIME, feature ranking) |

## ⚙️ Features Engineered

- Historical delinquency score trends
- Count of suspensions, penalty streaks, recent behavior stats
- Account age, customer type, normalized payment ratios

## 🧷 Output

- `features.csv`: Engineered features for each account snapshot
- `predictions.csv`: Prediction outputs including probabilities and risk labels
- SHAP and LIME values for interpretability

## 📘 Learn More

Full methodology, feature logic, and model analysis are documented in:
📄 `../reports/report_modeling.pdf`

---
