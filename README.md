# **Heart Disease Prediction - README**

## **Project Overview**
This project focuses on **predicting heart disease severity** using **Machine Learning (ML) and Deep Learning (DL)** techniques. The model is trained on the **UCI Heart Disease Dataset** and applies advanced preprocessing, hyperparameter tuning, and feature selection techniques to improve accuracy.

## **Dataset**
The dataset contains multiple medical attributes such as **age, cholesterol levels, chest pain type, blood pressure, exercise-induced angina, and more**. The target variable is a **multi-class classification label (0 - No Disease, 1 - Mild, 2 - Severe)**.

## **Models Implemented**
- **Machine Learning Models:**
  - Decision Tree Classifier
  - Support Vector Machine (SVM)
  - Logistic Regression
- **Deep Learning Models:**
  - Multi-Layer Perceptron (MLP)
  - Optimized Neural Network (Tuned MLP)

## **Key Techniques Used**
### **1. Data Preprocessing**
- Handling missing values using median imputation.
- Feature encoding (One-hot encoding for categorical variables).
- Min-max scaling for numerical features.

### **2. Feature Engineering & Selection**
- **Principal Component Analysis (PCA)** to reduce dimensionality while preserving variance.
- **Correlation Analysis** to remove redundant features.

### **3. Data Balancing**
- **Synthetic Minority Over-sampling Technique (SMOTE)** to address class imbalance.

### **4. Model Optimization**
- **Hyperparameter tuning** using Grid Search and Random Search.
- **Batch normalization & dropout layers** to prevent overfitting in neural networks.

## **Installation & Setup**
### **1. Clone the Repository**
```bash
git clone https://github.com/your-repository/heart-disease-prediction.git
cd heart-disease-prediction
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the Jupyter Notebook**
```bash
jupyter notebook Heart_disease_FP.ipynb
```

## **Usage**
- Open the Jupyter Notebook (`Heart_disease_FP.ipynb`).
- Run each cell sequentially to preprocess data, train models, and evaluate performance.
- Modify hyperparameters and dataset balancing techniques for further experimentation.

## **Results Summary**
| Model | Accuracy |
|--------|---------|
| Decision Tree | 71.01% |
| SVM | 68.12% |
| Logistic Regression | 71.74% |
| Baseline Neural Network | 73.12% |
| Tuned Neural Network | 78.34% |
| Optimized Neural Network | 82.15% |

## **Future Work**
- Implement **CNN and Transformer-based architectures** for further improvements.
- Apply **Explainable AI (XAI)** techniques to interpret model decisions.
- Explore **real-world deployment** in medical diagnosis applications.

## **Contributors**
- **Roei Aviv** (Afeka College, MSc in Intelligent Systems)
- **Omer Ben Simon** (Afeka College, MSc in Intelligent Systems)

## **License**
This project is licensed under the **MIT License**.

