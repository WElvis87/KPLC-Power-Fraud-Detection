# ⚡ KPLC Power Fraud Detection

## 🎯 Problem Definition: Why This Project Exists

Electric power theft and billing fraud are persistent challenges for utility companies worldwide, especially in emerging markets. Kenya Power Limited (KPLC) experiences significant non-technical losses due to unauthorized connections, meter tampering, irregular consumption patterns, and billing discrepancies. These activities reduce revenue collection, compromise grid reliability, and erode trust in the electricity network.

Traditional rule-based systems and manual audits are neither scalable nor adaptive to evolving fraudulent behavior. There is a need for a **data-driven machine learning system** that can detect abnormal electricity consumption patterns and support proactive intervention strategies.

The goal of this project is to build a fraud detection system that analyzes historical electricity consumption and related features to identify potential fraud or non-technical losses, prioritizing high-risk accounts for further investigation.

---

## 📌 Project Overview

**KPLC Power Fraud Detection** is a machine learning project that detects potential electricity fraud or non-technical loss using consumption patterns and engineered features from meter and account data.

This project demonstrates:
- Data ingestion, cleaning, and preprocessing  
- Feature engineering for consumption pattern analysis  
- Construction of machine learning models for fraud detection  
- Evaluation using metrics suited to highly imbalanced datasets

👉 The ultimate aim is to support utility analysts and KPLC stakeholders in identifying likely fraud cases early, reducing financial losses, and improving operational efficiency.

---

## 📊 Motivation

Electricity theft and fraudulent billing reduce revenue and impact service reliability across utility grids. Detecting these events:
- Saves financial losses for utilities  
- Enables targeted investigations and billing corrections  
- Improves grid stability and planning  
- Encourages equitable billing for customers

An effective detection system must deal with **imbalanced classes**, **noisy consumption data**, and evolving behavioral patterns.

---

## 🧠 What This Project Does:

This project builds a machine learning pipeline that:

1. Loads power consumption and customer data  
2. Cleans and engineers features that capture unusual patterns  
3. Trains classification models that distinguish normal vs suspicious accounts  
4. Evaluates models using robust metrics for imbalanced datasets  
5. Outputs fraud risk scores and flags for downstream review

It is a decision support tool — not a binary rule list.

---

### 🗂️ Project Structure

```
C:.
│   .gitignore
│   requirements.txt
│
├───config
│       config.yaml
│
├───data
│       kplc.csv
│       simulate.py
│
├───models
│       model1.kp1
│
├───notebooks
│       kplc.ipynb
│
├───reports
│       evaluation_report.md
│
├───src
│   │   config.py
│   │   data_cleaner.py
│   │   data_loader.py
│   │   data_splitting.py
│   │   pipeline.py
│   │   train_model.py
│   │   __init__.py
│   │
│   └───__pycache__
│           config.cpython-313.pyc
│           data_cleaner.cpython-313.pyc
│           data_loader.cpython-313.pyc
│           data_loading.cpython-313.pyc
│           data_splitting.cpython-313.pyc
│           load_data.cpython-313.pyc
│           train_model.cpython-313.pyc
│           __init__.cpython-313.pyc
│
└───tests
    │   test_config.py
    │   test_data_cleaning.py
    │   test_data_split.py
    │   test_load_data.py
    │   test_model_training.py
    │
    └───__pycache__
            test_load_data.cpython-313.pyc
            __init__.cpython-313.pyc
```

## 🧰 Dataset

### 🔒 Data Availability

Customer-level electricity consumption data from Kenya Power and Lighting Company (KPLC) is not publicly available due to privacy, security, and commercial sensitivity constraints. As a result, this project does not rely on proprietary or leaked data.
Instead, a synthetic dataset was generated to realistically approximate household electricity usage patterns and common electricity theft behaviors observed in power utilities.

### 🧪 Dataset Simulation Overview

The dataset represents monthly electricity consumption records for 2,000 households across 8 major Kenyan regions over a 3-year period (January 2022 – December 2024).

Each household is assigned:

1. A household type (low, medium, high consumption)
2. A geographic region
3. A baseline consumption profile

Consumption values are generated using:

1. Seasonal demand patterns
2. Household-specific base load
3. Random environmental noise
4. Weather-related variables (temperature and rainfall)

This approach ensures temporal continuity, behavioral consistency, and realistic variation in energy usage.

### 🌍 Geographic Coverage

The simulated regions include:

1. Nairobi
2. Mombasa
3. Kisumu
4. Nakuru
5. Eldoret
6. Nyeri
7. Thika
8. Malindi

These regions are included to reflect diverse urban, peri-urban, and coastal consumption patterns commonly observed in Kenya.

### 📐 Features Description

Each row in the dataset represents a household–month observation with the following features:

Feature	Description

household_id:	Unique household identifier
household_type:	Consumption category: low, medium, or high
region:	Geographic region
month:	Billing month
consumption_kWh:	Monthly electricity consumption
temperature:	Simulated average monthly temperature (°C)
rainfall:	Simulated monthly rainfall (mm)
prev_month_consumption:	Previous month’s consumption
consumption_diff:	Month-over-month consumption change
3m_avg:	Rolling 3-month average consumption
6m_avg:	Rolling 6-month average consumption
theft_flag:	Binary label indicating simulated electricity theft

### 🚨 Fraud Simulation Logic

Electricity theft is intentionally modeled as a rare event (~2%), consistent with real-world non-technical loss rates in power utilities.

Two types of theft behaviors are simulated:
1️⃣ Sudden Consumption Drops
Represents meter bypassing or abrupt tampering, where consumption is reduced sharply within a short time window.

2️⃣ Gradual Consumption Decline
Represents slow, progressive manipulation such as illegal tapping, where consumption decreases incrementally over time.

Households exhibiting either behavior are labeled with theft_flag = 1.

### ⚠️ Important Notes & Limitations

1. The dataset is fully synthetic and does not represent actual KPLC customers.
2. Fraud labels are simulated to support supervised learning experimentation.
3. Model results demonstrate method feasibility, not guaranteed real-world detection accuracy.
4. Real deployment would require validation on operational utility data.
5. The models can be directly applied to real KPLC consumption dataset

### 🎯 Purpose of the Dataset

The dataset is designed to:

1. Enable experimentation with fraud detection algorithms
2. Support temporal feature engineering
3. Simulate class imbalance challenges common in fraud detection
4. Provide a reproducible, ethical environment for model evaluation

### 🧠 Modeling Approach
Why Classification?
Electricity theft detection is framed as a binary classification problem:

0 → Normal consumption
1 → Suspected electricity theft

The task is to identify anomalous behavior indicative of fraud.

Why XGBoost?

1. XGBoost (Extreme Gradient Boosting) is selected as the primary model because it:
2. Handles non-linear feature interactions effectively
3. Performs well on tabular, structured data
4. Is robust to multicollinearity
5. Handles class imbalance better than linear models
6. Provides feature importance for interpretability
7. XGBoost is widely used in financial fraud detection, credit scoring, and risk modeling, making it a strong fit for this use case.

The column **isFraud** is used as the target value. The project uses the other columns to identify possibility of power fraud.

### 📏 Model Evaluation

The model is evaluated using metrics appropriate for imbalanced classification problems.

Primary Metrics
| Metric |	Why It Matters |
|--------|-----------------|
| Precision |	Minimizes false accusations |
| Recall |	Ensures fraud cases are not missed |
| F1-score |	Balances precision and recall |
| ROC-AUC |	Measures overall discrimination ability |
| PR-AUC |	More informative under class imbalance |


### 📈 Results Interpretation

Model outputs are risk scores and binary predictions, not final judgments.

Predictions are intended to:

1. Prioritize households for inspection
2. Reduce manual investigation workload
3. Improve targeting efficiency
4. A flagged account indicates higher likelihood of abnormal behavior, not definitive proof of theft.

## 🚀 Installation

### 🔧 Requirements  
Make sure Python (3.8+) is installed.

```
bash
# Clone repository
git clone https://github.com/WElvis87/KPLC-Power-Fraud-Detection.git
cd KPLC-Power-Fraud-Detection

# Create virtual environment
python -m venv venv
source venv/bin/activate     # Linux/macOS
venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
cd src
python pipeline.py
```
