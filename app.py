import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load Random Forest Classifier and StandardScaler
rf_classifier = joblib.load('random_forest_model.joblib')
feature_scaler = joblib.load('feature_scaler.joblib')
iris = load_iris()

# Streamlit App
st.title("Iris Flower Classification App")

# Input Fields for Features
sepal_length = st.text_input("Enter Sepal Length:", "0.0")
sepal_width = st.text_input("Enter Sepal Width:", "0.0")
petal_length = st.text_input("Enter Petal Length:", "0.0")
petal_width = st.text_input("Enter Petal Width:", "0.0")

# Convert input values to float
sepal_length = float(sepal_length)
sepal_width = float(sepal_width)
petal_length = float(petal_length)
petal_width = float(petal_width)

# Button to trigger prediction
if st.button("Predict"):
    # Preprocess input features
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_data_scaled = feature_scaler.transform(input_data)

    # Predictions using Random Forest
    rf_prediction = rf_classifier.predict(input_data_scaled)[0]

    # Display Predictions
    st.subheader("Random Forest Prediction:")
    st.write(f"Predicted Class (Random Forest): {iris.target_names[rf_prediction]}")
