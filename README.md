# Geometric Learning, Time-Variant Data Analysis, and Anomaly Detection

This repository contains the project implementations developed for the course *01HZSSM - Geometric Learning, Time-Variant Data analysis, and Anomaly Detection* at Politecnico di Torino.

## 📚 Course Overview

This advanced course explores geometric machine learning, graph neural networks (GNNs), and time-variant data analysis to detect anomalies in dynamic systems. It bridges non-Euclidean data representations (graphs, manifolds) with cutting-edge temporal architectures. Through real-world datasets, students gain practical expertise in building forecasting and anomaly detection pipelines for complex domains like finance and cybersecurity.

Key learning outcomes:
- **Geometric & Graph Learning:** Mathematical foundations, GCN, GAT, and Temporal GNNs.
- **Time-Series Modeling:** Classical (ARIMA, VAR) and deep learning methods (RNNs, LSTMs, TCNs).
- **Anomaly Detection:** Statistical tests, autoencoders, and hybrid models for temporal/graph data.

## 🗂️ Repository Structure

- **📄 `creditFraud.ipynb`**: The core Jupyter Notebook containing the end-to-end analytical pipeline. It covers exploratory data analysis, temporal feature engineering, and the training of supervised models (LogReg, LightGBM) versus label-free anomaly detectors (Isolation Forest, MixedAE, LSTM-AE, GRU-AE). 
- **📊 `creditFraud.pdf`**: The presentation summarizing the research and findings. It demonstrates how supervised models set a high ceiling in-distribution but fail under concept drift, while sequential autoencoders maintain robustness by learning a card-relative normality baseline.
