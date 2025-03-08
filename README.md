Enhancing Multi-Class Classification for Heart Disease Prediction

Authors: Roei Aviv & Omer Ben Simon

Affiliation: MSc in Intelligent Systems, Afeka College of Engineering, Tel Aviv, Israel


Project Overview
This repository contains the code, data, and documentation for a research project focused on improving multi-class classification for heart disease prediction using Machine Learning (ML) and Deep Learning (DL) techniques. The study compares traditional ML models—Decision Trees, Support Vector Machines (SVM), and Logistic Regression—with an optimized Multi-Layer Perceptron (MLP) Neural Network, leveraging the Heart Disease UCI Dataset. Key enhancements include Principal Component Analysis (PCA) for dimensionality reduction, Synthetic Minority Oversampling Technique (SMOTE) for class balancing, and hyperparameter tuning, achieving a peak accuracy of 82.15%.

Objectives
Assess baseline vs. optimized performance of ML and DL models.
Implement PCA and SMOTE to mitigate feature redundancy and class imbalance.
Develop a scalable, AI-driven framework for cardiovascular diagnostics.
Key Results
Optimized MLP: 82.15% accuracy, with F1-scores of 86% (Class 0), 79% (Class 1), and 80% (Class 2).
Baseline Comparison: Outperformed ML models (68–73% accuracy) by up to 14%.
Statistical Validation: Significant improvements confirmed via paired t-tests (e.g., MLP: p = 0.005).


Dataset
The project utilizes the Heart Disease UCI Dataset:

Size: 920 instances
Features: 13 predictive attributes, including:
Numerical: age, resting blood pressure (trestbps), cholesterol (chol), maximum heart rate (thalch), ST depression (oldpeak), number of major vessels (ca)
Categorical: sex, chest pain type (cp), fasting blood sugar (fbs), resting ECG (restecg), exercise-induced angina (exang), slope, thalassemia (thal)
Target: Multi-class label (0: No Disease, 1: Mild Disease, 2: Severe Disease)
Class Distribution: 411 (Class 0), 265 (Class 1), 109 (Class 2)
Source: UCI Machine Learning Repository
The dataset is provided in the data/ directory as heart_disease_uci.csv.

Prerequisites
Python Version: 3.8 or higher
Dependencies: Specified in requirements.txt. Key libraries include:
pandas (data manipulation)
scikit-learn (ML models, preprocessing, evaluation)
tensorflow or keras (Neural Networks)
imblearn (SMOTE implementation)
numpy (numerical operations)
matplotlib and seaborn (visualization)


Usage
Running the Analysis
Open notebooks/heart_disease_analysis.ipynb in Jupyter Notebook or JupyterLab to explore the dataset, train models, and visualize results.
Follow the notebook’s sequential steps:
Load and preprocess the dataset.
Train baseline models.
Apply PCA and SMOTE, then train optimized models.
Evaluate and compare performance metrics.
Preprocessing and Modeling
Preprocessing: Use src/preprocess.py to handle missing values, encode categorical variables, normalize data, and balance classes with SMOTE.
Modeling: Implement models via src/models.py, which includes Decision Trees, SVM, Logistic Regression, and MLP definitions along with training routines.
Optimization: Apply PCA and hyperparameter tuning with src/optimize.py.
Example Workflow
Load heart_disease_uci.csv and preprocess it (imputation, encoding, scaling, SMOTE).
Train baseline models and record performance.
Reduce dimensions with PCA, balance data with SMOTE, and retrain optimized models.
Generate metrics (accuracy, precision, recall, F1-score) and visualizations (e.g., confusion matrices).
Results Summary
Baseline Performance:
Decision Tree: 71.01% accuracy
SVM: 68.12% accuracy
Logistic Regression: 71.74% accuracy
MLP: 73.12% accuracy
Optimized Performance:
Decision Tree: 75.51% accuracy (F1: 78%, 72%, 73%)
SVM: 71.92% accuracy (F1: 75%, 68%, 70%)
Logistic Regression: 76.94% accuracy (F1: 79%, 74%, 75%)
MLP: 82.15% accuracy (F1: 86%, 79%, 80%)
Statistical Significance: Paired t-tests showed significant improvements (e.g., Decision Tree: p = 0.032; MLP: p = 0.005).
PCA Impact: Reduced features from 23 to 10, retaining 95% variance, cutting training time by 30%.
SMOTE Impact: Increased Class 2 recall from 58% to 79%, enhancing minority class detection.
Detailed results, including confusion matrices and statistical analyses, are available in docs/final_paper.md.

Contributing
We welcome contributions to enhance this project! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature-branch).
Commit your changes (git commit -m "Add feature X").
Push to the branch (git push origin feature-branch).
Open a Pull Request.
Please adhere to PEP 8 style guidelines and include comments or documentation for clarity.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
UCI Machine Learning Repository for providing the Heart Disease UCI Dataset.
Afeka College of Engineering for academic guidance and resources.
Open-Source Community for tools like scikit-learn, TensorFlow, and imblearn that made this work possible.
Contact
For inquiries, feedback, or collaboration opportunities, please contact:

