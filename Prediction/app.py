# Import the Libraries
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import os

# Load the trained model
model_file_path = 'Best_Random_Forest_Model.pkl'  

try:
    with open(model_file_path, 'rb') as f:
        model = pickle.load(f)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file not found. Please ensure the model file exists at the specified path.")
    st.stop()  # Stop execution if the model cannot be loaded
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()  # Stop execution for any other error

# Function to take user input
def get_user_input():
    Age = st.slider("Age", 0, 100, 50)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Smoking_Status = st.selectbox("Smoking Status", ["Former", "Current", "Never"])
    BMI = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    Air_pollution_Level = st.slider("Air Pollution Level", 0, 300, 100)
    Family_History_COPD = st.selectbox("Family History of COPD", [0, 1])
    COPD_Diagnosis = st.selectbox("Diagnosis of COPD", [0, 1])
    Age_Category= st.selectbox("Age categorry",["30-39", "40-49","50-59","60-69","70-79"])
    BMI_Category= st.selectbox("BMI categorry",["underweight", "normalweight","overweight","obesity"])
    Pollution_Risk_Score= st.selectbox("Pollution Risk Score",[0, 1])
    Smoking_Status_Encoded=st.selectbox("Smoking status ",[0, 1])
    Smoking_Pollution_interaction= st.number_input("Smoking Pollution interaction", min_value=10.0, max_value=50.0, value=25.0)
    
    # Handle Location input (example with one-hot encoding for 9 locations)
    location = st.selectbox("Location", [
        "Biratnagar", "Butwal", "Chitwan", "Dharan", 
        "Hetauda", "Kathmandu", "Lalitpur", 
        "Nepalgunj", "Pokhara"
    ])

    # Create one-hot encoded columns for location
    location_features = [1 if loc == location else 0 for loc in [
        "Biratnagar", "Butwal", "Chitwan", "Dharan", 
        "Hetauda", "Kathmandu", "Lalitpur", 
        "Nepalgunj", "Pokhara"
    ]]

    # Encode categorical variables
    gender_male = 1 if Gender == "Male" else 0
    smoking_former = 1 if Smoking_Status == "Former" else 0
    smoking_never = 1 if Smoking_Status == "Never" else 0
    smoking_current = 1 if Smoking_Status == "Current" else 0

    # Age bins (assuming 3 bins for simplicity)
    age_bin_adult = 1 if Age < 40 else 0
    age_bin_middle_aged = 1 if 40 <= Age < 60 else 0
    age_bin_elderly = 1 if Age >= 60 else 0

    # Log-transformed features (using simple log if applicable)
    air_pollution_log = np.log(Air_pollution_Level + 1)  # Prevent log(0)
    bmi_log = np.log(BMI)  # Assuming bmi won't be zero

    # Create input array
    input_data = np.array([[Age, Family_History_COPD, 
                            BMI, Air_pollution_Level, gender_male, smoking_former, 
                            smoking_never, smoking_current,COPD_Diagnosis,Age_Category ,BMI_Category,Pollution_Risk_Score *location_features,
                            Smoking_Status_Encoded,Smoking_Pollution_interaction,air_pollution_log, bmi_log,
                            age_bin_adult, age_bin_middle_aged, age_bin_elderly]])
    
    return input_data

# Main app
st.title("COPD Prediction")
st.write("Enter the patient's information to predict the likelihood of COPD.")

# Get user input
input_data = get_user_input()

# Make prediction
if st.button("Predict"):
    try:
        # Check if the model is fitted
        if not hasattr(model, "estimators_"):
            st.error("The model is not fitted. Please check the model file.")
            st.stop()  # Stop execution if the model isn't fitted
        
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("COPD Likely")
        else:
            st.success("COPD Not Likely")
    except Exception as e:
        st.error("An error occurred during prediction. Please check your inputs and try again.")
        st.error(f"Error details: {e}")
