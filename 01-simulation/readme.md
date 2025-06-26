# 🏗 Data Simulation – Synthetic Utility Billing Generator

This folder contains the codebase for generating a realistic utility billing dataset that mimics customer energy usage, billing cycles, payment behavior, and delinquency dynamics.

## 📌 What It Does

- Generates thousands of customers and accounts, each with unique reliability traits
- Simulates 72 months of billing, payment, and delinquency lifecycle
- Produces multi-table datasets reflecting real-world utility operations:
  - Customers, Accounts
  - Usages, Billings, Payments
  - Balance Snapshots, Bad Debt closures

## 🧱 Key Files

| File/Folder | Description |
|-------------|-------------|
| `simulate.py` | Main script to run the simulation and generate all tables |
| `config.yaml` | Customizable settings for simulation scale and behavioral assumptions |
| `data_generator/` | Contains modular code for agent behavior, billing logic, system checks |
| `output/` | Generated CSV files (or directs to dataset save path) |

## ⚙️ Configuration

Edit `config.yaml` to change:
- Number of customers / simulation cycles
- Usage ranges and plan rates
- Reliability and delinquency logic
- Escalation thresholds

## 🏁 Output Tables

The simulation outputs CSVs including:
- `customers.csv`
- `accounts.csv`
- `usages.csv`
- `billings.csv`
- `payments.csv`
- `balance_snapshots.csv`
- `bad_debts.csv`

## 📝 Read More

Full explanation, diagrams, and account case studies are available in:
📄 `../reports/report_simulation.pdf`

---
