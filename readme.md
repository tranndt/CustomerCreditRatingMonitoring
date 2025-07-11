# 🔍 Customer Credit Rating Classification & Monitoring – Project Overview

This repository contains a complete, end-to-end solution for simulating, analyzing, and monitoring **customer delinquency and credit risk** in a utility service billing context. It consists of three core components:

* 🏗 **Data Engineering** – Engineering realistic billing and delinquency data records using simulation
* 🧠 **Machine Learning** – Feature engineering and modeling to classify accounts into risk tiers
* 📊 **Power BI Dashboard** – Visual reporting to monitor account risk, delinquency progression, and operational KPIs


## 📝 Reports

Read the key reports here:

* [**Full Report - Customer Credit Rating Classification & Monitoring.pdf**](https://github.com/tranndt/CustomerCreditRatingMonitoring/blob/894de102cc59f4eb87245e9f58f16fb053710b2c/Full%20Report%20-%20Customer%20Credit%20Rating%20Classification%20%26%20Monitoring.pdf)
* [**Report #1 - Engineering A Utility Service Dataset.pdf**](https://github.com/tranndt/CustomerCreditRatingMonitoring/blob/894de102cc59f4eb87245e9f58f16fb053710b2c/01-data-engineering/Report%20%231%20-%20Engineering%20A%20Utility%20Service%20Dataset.pdf)
* [**Report #2 - Creating A Customer Credit Rating Classification System.pdf**](https://github.com/tranndt/CustomerCreditRatingMonitoring/blob/894de102cc59f4eb87245e9f58f16fb053710b2c/02-modelling/Report%20%232%20-%20Creating%20A%20Customer%20Credit%20Rating%20Classification%20System.pdf)
* [**Report #3 - Monitoring Customer Credit Risk in Power BI.pdf**](https://github.com/tranndt/CustomerCreditRatingMonitoring/blob/894de102cc59f4eb87245e9f58f16fb053710b2c/03-reporting-dashboard/Report%20%233%20-%20Monitoring%20Customer%20Credit%20Risk%20in%20Power%20BI.pdf)


## 📁 Repository Structure

| Folder                               | Description                                                                          |
| ------------------------------------ | ------------------------------------------------------------------------------------ |
| `01-data-engineering/`                | Python scripts and config files used to generate the full synthetic dataset          |
| `02-modelling/`           | Notebooks for feature engineering, labeling, model training, and results analysis    |
| `03-reporting-dashboard/`              | Supporting files, data outputs, and references for the Power BI monitoring dashboard |

---

## 📂 Dataset Access

The full datasets (CSVs) used in the project can be downloaded here:

**🔗 [Download Dataset (One Drive)](https://umanitoba-my.sharepoint.com/:f:/g/personal/tranndt_myumanitoba_ca/Enq4iqpaqPxDrBiVE27iDewBCOyi18MZwIZDwBHOZPdkjA?e=YxiCgC)**

Includes:

`data/01-data-engineering`

* Customer & Account info
* Usage, Billing & Payment history
* Balance Snapshots & Bad Debt records
* Engineered features & prediction outputs

`data/02-modelling`

* Raw balance snapshot data
* ML-ready snapshot features, labels & metadata
* Snapshot predictions and explanations

`data/03-reporting-dashboard`

* Raw balance snapshot data 
* Aggregated snapshot features
* Snapshot predictions and explanations


## 📄 Project Reports

To read a full explanation of each phase of the project, refer to the following reports:

| Report                  | Description                                                   |
| ----------------------- | ------------------------------------------------------------- |
| `Project Report.pdf`  | An overview of the entire project series, detailing how we      |
| `01-data-engineering/Part 1 - Data Engineering.pdf` | How the simulation works and case studies of account behavior |
| `02-modelling/Part 2 - Machine Learning.pdf`   | Feature creation, risk labeling approach, and model results        |
| `03-reporting-dashboard/Part 3 - Power BI Report.pdf`  | Final Power BI dashboard features and visual walkthroughs     |

---

## ✅ Getting Started

If you'd like to run the simulation or modeling locally:

1. Navigate to the corresponding folder (`01-data-engineering/` or `02-modelling/`)
2. Create a Python environment (`requirements.txt`)
3. Run notebooks or scripts as outlined in their README files


## 📬 Questions or Feedback?

Feel free to open an issue or reach out if you're curious about adapting this pipeline to your own industry or data!


