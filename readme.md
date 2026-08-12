# Adversarial Attacks Against a Random Forest-Based Electricity Theft Detector

## Overview

This project investigates the robustness of a Random Forest electricity theft detector against adversarial evasion attacks.

The detector was evaluated under:

- White-box attacks
- Gray-box attacks
- Black-box attacks

using domain-realistic perturbation strategies.

---

## Dataset

SGCC Electricity Theft Dataset

Features engineered include:

- Mean consumption
- Standard deviation
- Peak-to-mean ratio
- Rolling standard deviation
- Weekend usage
- Weekday usage
- Skewness
- Zero consumption percentage

---

## Workflow

1. Data preprocessing
2. Feature engineering
3. Class imbalance handling
4. Random Forest training
5. Hyperparameter tuning
6. Adversarial attack simulation
7. Performance evaluation

---

## Results

Baseline Random Forest

- ROC-AUC: 0.7463
- Theft Recall: 46%
- Theft Precision: 22%

Attack Success

- Gray-box: 58.7%
- White-box: 74.5%
- Black-box: 71.7%

---

## Technologies

Python

scikit-learn

NumPy

Pandas

Matplotlib

Jupyter Notebook

---

## Citation

B.Tech Final Year Project
Federal University of Technology Akure