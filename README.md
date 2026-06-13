# Geometric Learning, Time-Variant Data Analysis, and Anomaly Detection

This repository contains the project implementations developed for the course *01HZSSM - Geometric Learning, Time-Variant Data analysis, and Anomaly Detection* at Politecnico di Torino.

## 📚 Course Overview

The course explores the convergence of geometric machine learning, graph neural networks (GNNs), and time-variant data analysis, focusing on anomaly detection in dynamic systems. It bridges the theoretical foundations of non-Euclidean data representation with cutting-edge architectures like temporal and heterogeneous GNNs. Students learn to implement classic and deep learning approaches for time-series forecasting and distance- or density-based anomaly detection. Through real-world datasets, the curriculum develops practical expertise in building robust pipelines for complex domains such as cybersecurity and finance.

Key learning outcomes include:
- **Geometric & Graph Learning:** Mathematical foundations of manifolds and implementation of GCN, GAT, GIN, and Temporal GNNs.
- **Time-Series Modeling:** Classical methods (ARIMA, VAR, Kalman Filters) alongside deep learning architectures (RNNs, LSTMs, TCNs).
- **Anomaly Detection:** Statistical tests, autoencoders, and hybrid deep learning models specifically tailored for temporal and graph data.
- **Practical Implementation:** Developing and evaluating end-to-end pipelines using PyTorch Geometric, scikit-learn, and TensorFlow with time-aware metrics (PR-AUC, ROC-AUC).

## 🗂️ Repository Structure

This repository is dedicated to a comprehensive research project evaluating anomaly detection models against supervised baselines in the context of credit card fraud and concept drift:

- **📄 `creditFraud.ipynb`**: The core Jupyter Notebook containing the full analytical pipeline. It details the Exploratory Data Analysis, custom feature engineering (including causal expanding windows and cyclic temporal features), and the training of supervised models (Logistic Regression, LightGBM) versus label-free models (Isolation Forest, MixedAE, LSTM-AE, GRU-AE). The notebook culminates in a Phase 3 concept drift simulation.
- **📊 `creditFraud.pdf`**: The presentation summarizing the project's research question, methodology, and core findings. It highlights how supervised models (like LightGBM) set a high ceiling in-distribution but collapse under concept drift, whereas sequential autoencoders (LSTM-AE, GRU-AE) maintain robustness by learning a card-relative baseline of normality rather than a global one.
