import pandas as pd
import numpy as np

file_path = "dataset/diabetic_data.csv"

df = pd.read_csv(file_path)

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

# Dataset information
print("\nDataset Information:")
df.info()

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Missing Value Analysis

missing_values = df.isnull().sum()

missing_values = missing_values[missing_values > 0]

print("\nColumns with Missing Values:")
print(missing_values)

print("\nPercentage of Missing Values:")
print((df.isnull().sum() / len(df)) * 100)

# Replace '?' with NaN

df.replace('?', np.nan, inplace=True)

print("\nMissing Values After Replacement:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

# Duplicate Record Analysis

duplicates = df.duplicated().sum()

print("\nNumber of Duplicate Records:")
print(duplicates)

# Remove duplicate records
df.drop_duplicates(inplace=True)

print("\nDataset Shape After Removing Duplicates:")
print(df.shape)

# Target Variable Analysis

print("\nReadmission Distribution:")
print(df['readmitted'].value_counts())

print("\nReadmission Percentage:")
print(df['readmitted'].value_counts(normalize=True) * 100)

# Create Binary Target Variable

df['target'] = (df['readmitted'] == '<30').astype(int)

print("\nBinary Target Distribution:")
print(df['target'].value_counts())

print("\nTarget Percentage:")
print(df['target'].value_counts(normalize=True) * 100)

# Feature Engineering

# Total previous visits
df['total_previous_visits'] = (
    df['number_outpatient'] +
    df['number_emergency'] +
    df['number_inpatient']
)

# Total procedures and medications
df['total_treatment'] = (
    df['num_lab_procedures'] +
    df['num_procedures'] +
    df['num_medications']
)

# Hospital stay category
df['stay_category'] = pd.cut(
    df['time_in_hospital'],
    bins=[0, 3, 7, 14, 100],
    labels=['Short', 'Medium', 'Long', 'Very Long']
)

# Number of previous inpatient visits
df['previous_inpatient_visits'] = df['number_inpatient']

# Number of previous emergency visits
df['previous_emergency_visits'] = df['number_emergency']

# Number of previous outpatient visits
df['previous_outpatient_visits'] = df['number_outpatient']

print("\nNew Features Created:")
print(df[['total_previous_visits',
          'total_treatment',
          'stay_category',
          'previous_inpatient_visits',
          'previous_emergency_visits',
          'previous_outpatient_visits']].head())

print("\nUpdated Dataset Shape:")
print(df.shape)

# Prepare Data for Machine Learning

# Remove identifiers and target-related columns
X = df.drop(columns=[
    'encounter_id',
    'patient_nbr',
    'readmitted',
    'target'
])

# Target variable
y = df['target']

print("\nFeature Dataset Shape:")
print(X.shape)

print("\nTarget Dataset Shape:")
print(y.shape)

print("\nFeature Columns:")
print(X.columns.tolist())

# Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())



# Baseline Model - Logistic Regression

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object', 'category']).columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

# Process numerical data
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Process categorical data
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_columns),
    ('cat', categorical_pipeline, categorical_columns)
])

# Create model pipeline
baseline_model = Pipeline([
    ('preprocessing', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

# Train model
baseline_model.fit(X_train, y_train)

# Predict
y_pred = baseline_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nBaseline Model Accuracy:")
print(accuracy)

# Baseline Model Evaluation

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

y_prob = baseline_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

print("\n===== Baseline Model Evaluation =====")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Random Forest Model

from sklearn.ensemble import RandomForestClassifier

random_forest_model = Pipeline([
    ('preprocessing', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ))
])

# Train model
random_forest_model.fit(X_train, y_train)

# Make predictions
rf_predictions = random_forest_model.predict(X_test)

# Evaluate accuracy
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\n===== Random Forest Model =====")
print("Accuracy:", rf_accuracy)

print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))

# Model Comparison

comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, rf_predictions)
    ],
    'Precision': [
        precision_score(y_test, y_pred, zero_division=0),
        precision_score(y_test, rf_predictions, zero_division=0)
    ],
    'Recall': [
        recall_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, rf_predictions, zero_division=0)
    ],
    'F1 Score': [
        f1_score(y_test, y_pred, zero_division=0),
        f1_score(y_test, rf_predictions, zero_division=0)
    ]
})

print("\n===== MODEL COMPARISON =====")
print(comparison.round(4))

# Improved Logistic Regression with Class Balancing

balanced_model = Pipeline([
    ('preprocessing', preprocessor),
    ('model', LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    ))
])

# Train
balanced_model.fit(X_train, y_train)

# Predict
balanced_predictions = balanced_model.predict(X_test)

# Evaluate
print("\n===== Balanced Logistic Regression =====")

print("Accuracy:", accuracy_score(y_test, balanced_predictions))

print("Precision:", precision_score(
    y_test, balanced_predictions, zero_division=0
))

print("Recall:", recall_score(
    y_test, balanced_predictions, zero_division=0
))

print("F1 Score:", f1_score(
    y_test, balanced_predictions, zero_division=0
))

print("\nClassification Report:")
print(classification_report(
    y_test, balanced_predictions, zero_division=0
))

# Final Model Comparison

final_comparison = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'Random Forest',
        'Balanced Logistic Regression'
    ],
    'Accuracy': [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, rf_predictions),
        accuracy_score(y_test, balanced_predictions)
    ],
    'Precision': [
        precision_score(y_test, y_pred, zero_division=0),
        precision_score(y_test, rf_predictions, zero_division=0),
        precision_score(y_test, balanced_predictions, zero_division=0)
    ],
    'Recall': [
        recall_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, rf_predictions, zero_division=0),
        recall_score(y_test, balanced_predictions, zero_division=0)
    ],
    'F1 Score': [
        f1_score(y_test, y_pred, zero_division=0),
        f1_score(y_test, rf_predictions, zero_division=0),
        f1_score(y_test, balanced_predictions, zero_division=0)
    ]
})

print("\n===== FINAL MODEL COMPARISON =====")
print(final_comparison.round(4))

import matplotlib.pyplot as plt

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

plt.figure(figsize=(10, 6))

for model_name in final_comparison['Model']:
    values = final_comparison.loc[
        final_comparison['Model'] == model_name, metrics
    ].values.flatten()

    plt.plot(metrics, values, marker='o', label=model_name)

plt.title('Model Performance Comparison')
plt.xlabel('Evaluation Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Step 20: Threshold Adjustment

y_prob_balanced = balanced_model.predict_proba(X_test)[:, 1]

threshold = 0.3

threshold_predictions = (y_prob_balanced >= threshold).astype(int)

print("\n===== THRESHOLD ADJUSTMENT =====")
print("Threshold:", threshold)

print("Accuracy:", accuracy_score(y_test, threshold_predictions))
print("Precision:", precision_score(y_test, threshold_predictions, zero_division=0))
print("Recall:", recall_score(y_test, threshold_predictions, zero_division=0))
print("F1 Score:", f1_score(y_test, threshold_predictions, zero_division=0))

# Step 21: Confusion Matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, threshold_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Not Readmitted', 'Readmitted']
)

disp.plot(cmap='Blues')

plt.title('Confusion Matrix - Balanced Logistic Regression')
plt.tight_layout()
plt.show()

print("\nConfusion Matrix:")
print(cm)

# Step 22: Feature Importance

rf_classifier = random_forest_model.named_steps['model']

feature_names = random_forest_model.named_steps['preprocessing'].get_feature_names_out()

importance = rf_classifier.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance', ascending=False
)

print("\n===== TOP 15 IMPORTANT FEATURES =====")
print(feature_importance.head(15))

# Plot top 15 features
top_features = feature_importance.head(15)

plt.figure(figsize=(10, 7))
plt.barh(top_features['Feature'], top_features['Importance'])
plt.gca().invert_yaxis()

plt.title('Top 15 Features for Readmission Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Features')

plt.tight_layout()
plt.show()

# Step 23: Save Trained Model

import joblib
import os

os.makedirs("model", exist_ok=True)

joblib.dump(balanced_model, "model/readmission_model.pkl")

print("\nModel saved successfully!")
print("Saved location: model/readmission_model.pkl")

# Step 24: Prediction for a New Patient

import joblib

loaded_model = joblib.load("model/readmission_model.pkl")

new_patient = X_test.iloc[[0]]

prediction = loaded_model.predict(new_patient)
probability = loaded_model.predict_proba(new_patient)

print("\n===== NEW PATIENT PREDICTION =====")

if prediction[0] == 1:
    print("Prediction: Patient is at risk of readmission within 30 days")
else:
    print("Prediction: Patient is unlikely to be readmitted within 30 days")

print("Readmission Probability:", round(probability[0][1] * 100, 2), "%")

# Step 25: Reusable Prediction Function

def predict_readmission(patient_data):

    prediction = loaded_model.predict(patient_data)
    probability = loaded_model.predict_proba(patient_data)[0][1]

    if prediction[0] == 1:
        result = "High Risk of Readmission"
    else:
        result = "Low Risk of Readmission"

    print("\n===== PATIENT PREDICTION =====")
    print("Result:", result)
    print("Readmission Probability:", round(probability * 100, 2), "%")

    return result


# Test the function
sample_patient = X_test.iloc[[5]]

predict_readmission(sample_patient)