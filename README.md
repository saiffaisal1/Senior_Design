# Smart Microgrid Anomaly Detection using Isolation Forest 

This repository contains a complete machine learning pipeline for unsupervised anomaly detection in smart microgrid systems. The model identifies unusual voltage and frequency behaviors using Isolation Forest, helping improve the autonomy and resilience of microgrid controllers.

## Project Overview

The goal is to detect voltage and frequency anomalies in real-time to support autonomous control in microgrid environments. This proof-of-concept AI model uses historical data, simulated anomalies, and a hybrid visual reporting interface to track abnormal activity across energy distribution components.

---

## Directory Structure
```bash
├── Data/ # Raw and simulated CSV datasets
├── model/ # Trained Isolation Forest model and scaler (.pkl)
├── alerts/ # Detected anomaly reports (CSV)
├── monitoring/
│ ├── daily_monitor.py # Detect and export anomalies from new data
│ └── visualizer.py # Visualize anomalies in Plotly dashboards
├── Training/
│ ├── combine_data.py # combine all the CSVs into one big CSV
│ └── train_model.py # Train and save Isolation Forest model
└── README.md
```
## Key Features

-  Unsupervised anomaly detection using Isolation Forest
-  Modular and reusable training and monitoring scripts
-  Plotly dashboards for visual inspection of voltage & frequency anomalies
-  Confidence scoring for prioritizing alerts
-  Supports historical and real-time anomaly detection

---

## Installation

```bash
git clone https://github.com/saiffaisal1/Senior_Design
cd microgrid-anomaly-detector
pip install -r requirements.txt
