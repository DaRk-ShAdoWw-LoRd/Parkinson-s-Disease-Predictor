# Parkinson's Disease Predictor

A Machine Learning based web application for predicting the likelihood of Parkinson’s disease using voice measurement features.

## Application Link:  
https://parkinson-s-disease-predictor.streamlit.app/

---

# Project Overview

This project uses ensemble machine learning techniques to predict Parkinson’s disease from biomedical voice measurements.The workflow includes data preprocessing,feature engineering, hyperparameter tuning,ensemble learning,and deployment through Streamlit.

The final deployed model is a Soft Voting Ensemble consisting of:

1. Support Vector Classifier (SVC)
2. XGBoost Classifier
3. Random Forest Classifier

The project was developed with a strong focus on minimizing false negatives due to the medical nature of the prediction task.


# Features

## Data Preprocessing
- Feature renaming for readability
- Skewness correction using `PowerTransformer`
- Feature scaling using `StandardScaler`
- Selective preprocessing of skewed features using `ColumnTransformer`
- Pipeline integration to ensure consistent preprocessing during training and inference

## Machine Learning
- Hyperparameter tuning using `GridSearchCV`
- Stratified cross validation
- Ensemble learning using Soft Voting
- Comparison between Voting and Stacking classifiers
- Performance evaluation using:
  - Accuracy
  - F1 Score
  - Precision
  - Recall
  - Confusion Matrix

## Web Application
- Interactive Streamlit interface
- Real time prediction
- Confidence score display
- Example preset inputs
- Probability visualization

---

# Final Model Performance

## Ensemble Model
Soft Voting Ensemble Classifier

## Test Accuracy
**94.87%**

## Confusion Matrix Metrics

| Metric | Value |
|---|---|
| True Positives | 28 |
| True Negatives | 9 |
| False Positives | 1 |
| False Negatives | 1 |

The model achieved only one false negative, which was an important objective for this healthcare related prediction task.

---

# Individual Model Cross Validation Scores

| Model | Score |
|---|---|
| SVC | 0.956094 |
| XGBoostClassifier | 0.942284 |
| RandomForestClassifier | 0.940991 |
| LogisticRegression | 0.888181 |

Logistic Regression was excluded from the final ensemble due it's poor performance.

---

# Technologies Used

## Programming Language
- Python

## Libraries
- Scikit Learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

# Machine Learning Workflow

## Data Exploration
- Correlation analysis
- Histogram visualization
- Scatter matrix visualization

## Feature Engineering
- Created additional derived features(like Average Fundamental Frequence)
- Removed highly redundant attributes

## Model Training
The following models were trained and tuned:
- Support Vector Classifier
- Random Forest
- XGBoost
- Logistic Regression

## Ensemble Methods
The following ensemble methods were tested:
- Soft Voting Classifier
- Stacking Classifier

Soft voting with equal weighting produced the best performance after proper pipeline integration.

---
# Project Structure

```text
Parkinsons-Predictor/
│
├── app.py
├── Parkinson_Predictor_Model.pkl
├── requirements.txt
├── README.md
├── .gitignore
│
├── notebook/
│   └── Parkinson_Predictor.ipynb
│
└── data/
    └── parkinsons.data

```

---

# Important Notes

This project is intended for educational purposes only.

The application is not a substitute for professional medical diagnosis or clinical evaluation.

---

# Future Improvements

- Larger dataset for improved generalization
- Better UI and visualization features
- Cloud database integration
- User authentication and report generation

---

# Author

Developed by Binish Hari B
