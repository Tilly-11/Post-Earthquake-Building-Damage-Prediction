# 🏢 Post-Earthquake Building Damage Prediction

An end-to-end machine learning pipeline built with Scikit-Learn to predict structural damage levels in the Kavrepalanchok district (District 3) following a seismic event. This project demonstrates realistic target variable construction, exploratory data analysis, cost-sensitive learning for high class imbalance, and tree-based validation tuning.

## 📊 Project Overview

In post-disaster scenarios, class distributions are inherently heavily skewed—most structures near an epicenter sustain severe damage, while only a small subset remain structurally stable. This repository models that realistic class imbalance and trains predictive classifiers that remain highly sensitive to the minority class (undamaged buildings).

### Key Features
* **Robust Data Cleaning:** Automated loading and filtering of structural features with full leakage control (excluding post-earthquake attributes).
* **Imbalance Simulation:** Models an 87% (Severe Damage) vs. 13% (Undamaged) dataset split.
* **Cost-Sensitive Classifiers:** Implementation of Logistic Regression and Decision Trees using class balancing weights (`class_weight="balanced"`).
* **Hyperparameter Tuning:** Empirical max depth selection using validation curve plots.

---

## 📈 Results and Deliverables

### Class Imbalance Mitigation
By utilizing cost-sensitive weight adjustments, our models achieved exceptional sensitivity to undamaged structures, protecting the classification from defaulting to the majority class:
* **Logistic Regression (Balanced):** Achieved an overall validation accuracy of **95.9%** and a minority-class (undamaged) recall rate of **98%**.
* **Decision Tree Validation:** Tuning identified `max_depth=5` as the ideal threshold to maximize classification generalization without overfitting.

### Feature Importance
Based on Gini importance metrics from the tuned Decision Tree model, **`foundation_type`** emerged as the primary predictive indicator of a building's susceptibility to damage, followed by the material configuration of its **`superstructure`**.

---

## 🛠️ Project Structure

```text
├── data/
│   └── Nepal_dataset.csv       # Dataset (raw structural metrics)
├── notebooks/
│   └── damage_prediction.ipynb # Interactive Jupyter Notebook
├── requirements.txt            # Package dependencies
├── .gitignore                  # Python repository filters
└── README.md                   # Project documentation
