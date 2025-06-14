# 🧠 Income Classification using Machine Learning

This project is a complete end-to-end machine learning pipeline built using the **Adult Income Dataset** (UCI Repository). The goal is to predict whether an individual earns more than $50K per year based on attributes like age, education, occupation, and more.

---

## 🔍 Problem Statement

The task is a **binary classification**:
> Predict whether income >50K or <=50K using census data.

This dataset is widely used to benchmark ML algorithms on:
- Handling categorical features
- Dealing with class imbalance
- Feature engineering
- Model tuning

---

## 🚀 What This Project Covers

✅ End-to-end ML pipeline  
✅ Exploratory Data Analysis (EDA)  
✅ Feature Engineering  
✅ Handling missing values with `SimpleImputer`  
✅ Encoding categorical variables with `OneHotEncoder`    
✅ Dealing with class imbalance using `SMOTE`  
✅ Model building with `LGBMClassifier`  
✅ Hyperparameter tuning with `GridSearchCV`  
✅ 10-fold Stratified Cross-Validation  
✅ Accuracy: ~83%

---

## 🛠️ Tech Stack & Tools

- Python
- Pandas, NumPy
- Matplotlib, Seaborn (for basic EDA)
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Jupyter Notebook

---

## 🧱 Pipeline Structure

```python
# Preprocessing
- Missing value imputation
- One-hot encoding

# Oversampling
- SMOTE (for imbalanced classes)

# Model
- LGBMClassifier

# Tuning
- GridSearchCV with stratified K-Fold
