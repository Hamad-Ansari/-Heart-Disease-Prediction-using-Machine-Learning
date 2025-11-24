# 🫀 Heart Disease Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange)
![ML](https://img.shields.io/badge/Machine-Learning-brightgreen)
![Health](https://img.shields.io/badge/Domain-Healthcare%20AI-lightgrey)

A comprehensive machine learning project for predicting heart disease risk using clinical parameters. This project demonstrates end-to-end ML pipeline development from data analysis to deployment-ready prediction system.

## 📊 Project Highlights

- **AUC Score**: 0.92+ on test data
- **Accuracy**: 85%+ in predicting heart disease
- **Features**: 13 clinical parameters including age, cholesterol, blood pressure
- **Model**: Ensemble Random Forest with hyperparameter tuning
- **Dataset**: UCI Heart Disease Dataset (900+ patients)

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/Heart-Disease-Prediction-ML.git
cd Heart-Disease-Prediction-ML
pip install -r requirements.txt
```


## 🔧 Features
- Data Preprocessing: Handling missing values, feature engineering

- Multiple Algorithms: Comparison of 4+ ML models

- Hyperparameter Tuning: GridSearchCV for optimization

- Model Interpretation: Feature importance analysis

- Production Ready: Serialized model and prediction API

- Comprehensive Testing: Unit tests for all components

## 📚 Dataset
- The project uses the UCI Heart Disease Dataset containing:

- 920 patient records

- 13 clinical features

- 4 different datasets merged

Target: Presence of heart disease (0: No, 1: Yes)

Basic Prediction
```python

result = predictor.quick_predict(
    age=45,
    sex='female', 
    chest_pain_type='typical angina',
    cholesterol=180,
    blood_pressure=120
)
patients = [
    {'age': 45, 'sex': 'female', 'chol': 180, ...},
    {'age': 65, 'sex': 'male', 'chol': 280, ...}
]
results = predictor.predict_batch(patients)

````
