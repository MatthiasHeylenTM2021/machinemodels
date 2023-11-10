import streamlit as st
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

# Fetch the "Predict Students' Dropout and Academic Success" dataset from the UCI repository
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

# Data (as pandas DataFrames)
X = predict_students_dropout_and_academic_success.data.features
y = predict_students_dropout_and_academic_success.data.targets

# Convert string labels to numeric using Label Encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create a Logistic Regression classifier
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred_lr = logistic_regression.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

# Create an XGBoost classifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

# Create an SVM classifier
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

st.title("Artificial Intelligence Task 2")

# Streamlit interface for displaying results
st.write("Logistic Regression Accuracy:", f"{accuracy_lr:.2f}")
st.write("XGBoost Accuracy:", f"{accuracy_xgb:.2f}")
st.write("SVM Accuracy:", f"{accuracy_svm:.2f}")
