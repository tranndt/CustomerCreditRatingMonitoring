# 🏗 Data Engineering – Synthetic Utility Billing Simulation

This folder contains the simulation engine and configuration files used to create a realistic, multi-year dataset of utility customer billing and delinquency behavior.

## 🔧 What It Does

This simulation generates synthetic records for utility consumption, billing, payments, and delinquency events over a configurable timeline. It models realistic customer behavior using a mix of rules-based and probabilistic logic.

Key features include:

* Simulating customer account creation and **lifecycle behavior**
* Modeling **monthly usage**, **billing generation**, and **payment patterns**
* Tracking **delinquency progression** with penalty scoring
* Automatically **escalating account status** (Delinquent → Suspended → Closed)
* Producing a **comprehensive**, **multi-table dataset** for **downstream ML and dashboard use**

## 📂 Key Files

| File/Folder                     | Description                                                                         |
| ------------------------------- | ----------------------------------------------------------------------------------- |
| `Part 1 - Data Engineering.pdf` | Full report on simulation logic and account behavior case studies                   |
| `data_engineering.ipynb`        | Demo notebook for exploring and previewing the dataset                              |
| `simulation/simulate.py`        | Main script to generate the full dataset                                            |
| `simulation/config.yaml`        | Config file defining simulation parameters (e.g., cycle length, penalty thresholds) |

## ▶️ Running the Simulation

To generate the dataset locally:

```bash
# Create and activate your environment
pip install -r requirements.txt

# Run the simulation
python simulation/simulate.py
```

The outputs will be saved to the `data/` directory by default.

## 🧾 Output Tables

The following CSV files will be generated:

* `customers.csv`, `accounts.csv` – Customer and account metadata
* `usages.csv`, `billings.csv`, `payments.csv` – Monthly activity logs
* `balance_snapshots.csv` – Monthly snapshots of account state (balances, delinquency, penalties)
* `bad_debts.csv` – Records of accounts permanently closed due to high delinquency scores

These outputs provide the raw foundation for machine learning modeling and Power BI dashboard reporting.

## 📊 Demo Dataset

The pre-generated version of the dataset is available here:

**🔗 [Demo Dataset](https://umanitoba-my.sharepoint.com/:f:/g/personal/tranndt_myumanitoba_ca/EjdT412QMgBOsgIzCtkaAxwBbJ3U2IsTR-0lE6Bl3ZiTGw?e=gW8kWg)**

The demo dataset simulate the activity of a customer base of 3,000 customers over 72 billing cycles (6 years). 
By the end of the simulation, we generated: 

- **14,000 accounts** created 
- **580,000 records**each for usage, billings, and payments 
- **Full balance snapshot history** tracking financial state for every account, every month 
- **500+ accounts ultimately closed** and written off due to unresolved delinquency 


## 📘 Read More

For more background on the simulation logic, design considerations, and behavior examples:

📄 `Part 1 - Data Engineering.pdf`