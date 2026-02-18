import pandas as pd

# Load dataset
df = pd.read_csv("data/student-mat.csv", sep=';')

print("Dataset Loaded Successfully\n")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())


#step-2

import seaborn as sns
import matplotlib.pyplot as plt

# Check correlation for numerical columns
numeric_df = df.select_dtypes(include=['int64'])

plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

#step-3

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Select numeric features only
numeric_df = df.select_dtypes(include=['int64'])

# Define features and target
X = numeric_df.drop(["G3", "G1", "G2", "Medu", "Fedu", "famrel"], axis=1)
y = numeric_df["G3"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Results:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)


#step-4

from sklearn.ensemble import RandomForestRegressor

# Train Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predict
rf_pred = rf_model.predict(X_test)

# Evaluate
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Results:")
print("Mean Squared Error:", rf_mse)
print("R2 Score:", rf_r2)


#step-5

import numpy as np

# Feature importance
importances = rf_model.feature_importances_
feature_names = X.columns

# Sort importance
indices = np.argsort(importances)[::-1]

print("\nFeature Importance (Top 10):")
for i in range(10):
    print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")


#step-6

# Create classification target
df["pass"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

print("\nPass/Fail Distribution:")
print(df["pass"].value_counts())


#step-7  compare logistic vs randomforest

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Features (without G1, G2 , medu, fedu, famrel)
X_class = numeric_df.drop(["G3", "G1", "G2", "Medu", "Fedu", "famrel"], axis=1)
y_class = df["pass"]
print("Number of features used for training:", X.shape[1])


# Train-test split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(X_train_c, y_train_c)
log_pred = log_model.predict(X_test_c)

print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test_c, log_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test_c, log_pred))
print("Classification Report:\n", classification_report(y_test_c, log_pred))

# Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_c, y_train_c)
rf_pred_c = rf_classifier.predict(X_test_c)

print("\nRandom Forest Classifier Results:")
print("Accuracy:", accuracy_score(y_test_c, rf_pred_c))
print("Confusion Matrix:\n", confusion_matrix(y_test_c, rf_pred_c))
print("Classification Report:\n", classification_report(y_test_c, rf_pred_c))


# phase-2 save models professionally


import os
import joblib

# Create models folder if not exists
if not os.path.exists("models"):
    os.makedirs("models")

# Save models
joblib.dump(rf_model, "models/regression_model.pkl")
joblib.dump(log_model, "models/classification_model.pkl")

print("\nModels saved successfully inside /models folder.")

