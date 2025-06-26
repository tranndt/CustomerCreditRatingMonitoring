# 🧠 Machine Learning – Customer Credit Risk Classification

This folder contains all notebooks and scripts used to engineer features, define risk labels, and train a supervised machine learning model to classify utility customer accounts by credit risk.

## 🎯 What It Does

This phase builds on the synthetic dataset from Part 1 to identify which accounts are at risk of becoming delinquent or being closed due to nonpayment.

Key tasks include:

* Creating **behavioral features** that summarize each account’s history
* Designing a **three-tier risk labeling strategy** to distinguish low, medium, and high-risk accounts
* Training a **Random Forest model** to classify accounts into risk groups
* Generating **prediction probabilities and SHAP explanations** to support interpretability

## 📂 Key Files

| File/Folder                     | Description                                                               |
| ------------------------------- | ------------------------------------------------------------------------- |
| `Part 2 - Machine Learning.pdf` | Full report covering feature logic, label strategy, and model performance |
| `modelling_report.ipynb`        | Notebook demonstrating model outputs and case studies                     |
| `pipeline.ipynb`                | End-to-end walkthrough of the ML pipeline: feature creation to prediction |
| `pipeline/run_pipeline.py`      | Main script to run the full pipeline                                      |
| `pipeline/pipeline_utils.py`    | Utility functions used throughout the pipeline                            |

## 📥 Input Data

Located in `data/raw/`:

* `balance_snapshots.csv` – Raw monthly account state data
* `accounts.csv`, `bad_debts.csv` – Metadata from Part 1

You can generate these from `01-data-engineering/` or download them from:

**🔗 [Demo Dataset](https://umanitoba-my.sharepoint.com/:f:/g/personal/tranndt_myumanitoba_ca/El2HkuodHShOmfMw5a5JZ8cBqGlAQ0BOJlLP58izg_7lqQ?e=v7z3J3)**

**Dataset Overview:**

* 3,000 customers over 72 billing cycles (6 years)
* \~14,000 accounts created
* \~580,000 records for usage, billing, and payments
* 500+ accounts closed due to delinquency

## ▶️ Running the Pipeline

1. Install required packages:

```bash
pip install -r requirements.txt
```

2. Run the pipeline:

```bash
python pipeline/run_pipeline.py
```

All outputs will be saved under the `data/` folder.

## 🏗 Outputs

`data/preprocessed/`:

* `balance_snapshots_raw_features.csv` – Full aggregated behavioral history
* `balance_snapshots_features.csv` – Final training-ready features
* `balance_snapshots_metadata.csv` – Risk labels and useful metadata

`data/ml_output/`:

* `balance_snapshots_predictions.csv` – Risk class and probabilities
* `balance_snapshots_predictions_explanations.csv` – SHAP-based explanation for top N features per prediction

These outputs feed directly into the Power BI dashboard (Part 3) for interactive monitoring and account-level insights.

## 🧠 ML Summary

* **Model**: Random Forest classifier
* **Labels**:

  * A = Low Risk
  * B = Medium Risk
  * C = High Risk (Bad Debt)
* **Evaluation**: Classification report, precision/recall, SHAP feature importance

## 📘 Read More

📄 `Part 2 - Machine Learning.pdf` — includes full breakdown of the ML approach, case studies, and detailed explanation of feature engineering and label design.