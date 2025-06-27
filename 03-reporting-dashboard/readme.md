# 📊 Power BI – Delinquency Risk Monitoring Dashboard

This folder contains the Power BI dashboard used to visualize risk classification outputs and operational KPIs derived from simulated utility billing data and machine learning predictions.

## 📌 What It Does

This report enables real-time monitoring and decision support for customer delinquency risk. Built on the synthetic data and predictions from Parts 1 and 2, it helps organizations:

* Track key performance indicators (KPIs) such as bad debt trends and portfolio health
* Classify and visualize account risk levels (A, B, C) using explainable ML outputs
* Drill down into account-level behavior for analysis or operational action
* Identify patterns in energy usage, payments, penalties, and risk over time

## 🧪 Live Demo

Explore the interactive dashboard here:

🔗 [Power BI Online Demo](https://app.powerbi.com/view?r=eyJrIjoiOGVkYzY5MTAtMTU1Mi00Zjc3LThkNTctOTc1OWFlYjlmNWM2IiwidCI6IjZkMzI1MGEzLTg5NDUtNDNjZS05Nzg0LTlmMjcwZWVjYzQ1MSJ9)

Perfect for demos, interviews, or collaborative presentations.

## 📁 Folder Contents

| File/Folder                                             | Description                                                       |
| ------------------------------------------------------- | ----------------------------------------------------------------- |
| `Part 3 - Power BI Report.pdf`                          | Full documentation with dashboard walkthrough and design insights |
| `Utility Billing Delinquency Dashboard & Analysis.pbix` | Power BI file with data model, visuals, and logic                 |
| `data/`                                                 | Cleaned features and prediction data used by the dashboard        |

## 📥 Input Data

The dashboard reads from the following files in the `data/` folder:

* `accounts.csv`, `customers.csv`, `balance_snapshots.csv`
* `balance_snapshots_raw_features.csv` – Aggregated behavioral history
* `balance_snapshots_features.csv` – Normalized ML feature set
* `balance_snapshots_metadata.csv` – Snapshot status and risk labels
* `balance_snapshots_predictions.csv` – Predicted class and probabilities
* `balance_snapshots_predictions_explanations.csv` – SHAP explanations for model decisions

You can download the demo dataset from here:

**🔗 [Demo Dataset](https://umanitoba-my.sharepoint.com/:f:/g/personal/tranndt_myumanitoba_ca/Elecx3l-g2FEtkU9YJ3ZBR4Bbgj1UlpOrg_MbsHpoZhcaA?e=CdRXnK)**

These are produced in **Part 2 – Machine Learning**.

## 🧭 Dashboard Sections

* **Overview Dashboard** – High-level metrics for executive insight
* **Account Explorer** – Tabular explorer with filters and key account summaries
* **Account In-Depth Analysis** – Detailed drill-downs for selected accounts, including:

  * Assigned risk tier and explanation
  * Historical billing and payment timeline
  * Monthly delinquency score and status progression

## 🧠 Audience

* 📈 **Business Analysts** – Analyze risk segments and behavioral trends
* 💼 **Operations Teams** – Pinpoint at-risk accounts and support recovery efforts
* 🎯 **Executives** – Monitor portfolio-level exposure and performance

## 📘 Read More

📄 `Part 3 - Power BI Report.pdf` — includes a full visual walkthrough, rationale behind each section, and case studies highlighting usage scenarios.
