🔬 Algorithmic Stability Analysis of Machine Learning Models
📌 Overview

Machine learning models are commonly evaluated using predictive accuracy; however, accuracy alone does not capture how reliable a model is under small changes in training data. This project focuses on algorithmic stability, a key concept in statistical learning theory that measures the sensitivity of model predictions to minor data perturbations and is closely related to generalization behavior.

This study provides a systematic, empirical comparison of multiple machine learning algorithms by jointly analyzing accuracy and stability.

🎯 Objectives

Quantify algorithmic stability of machine learning models using prediction variance.

Analyze the accuracy–stability trade-off across different model families.

Study model robustness under controlled data perturbations.

Validate findings using both synthetic datasets (with known ground truth) and real-world data.

📊 Datasets
1️⃣ Synthetic Regression Dataset

Generated using a controlled data-generating process.

Known ground truth coefficients.

Allows precise control over noise and feature relevance.

Used for theoretical analysis of model stability.

2️⃣ Real-World Dataset: California Housing

Public benchmark dataset.

Realistic noise and feature correlations.

Used for empirical validation of stability behavior.

This dual-dataset approach enables both theoretical rigor and practical relevance.

🧠 Models Evaluated

The project compares models with different inductive biases:

Linear Regression

Random Forest Regressor

Support Vector Regression (SVR)

Neural Network (MLP with early stopping)

🔄 Stability Evaluation Methodology
Data Perturbation Techniques

Each model is trained multiple times using slightly modified training sets created via:

Bootstrap resampling

Random sample removal

Gaussian noise injection

Stability Metric

Stability is quantified as the average variance of predictions on a fixed test set across repeated trainings:

Lower values indicate higher stability.

📈 Performance Metrics

RMSE (Root Mean Squared Error) for predictive accuracy

Prediction variance for algorithmic stability

Key Insight:
Models with higher expressive power tend to achieve better accuracy but exhibit reduced stability, highlighting an important trade-off between performance and robustness.
