# Import the Libraries
import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the trained model
with open('Best_Random_Forest_Model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to take user input
def get_user_input():
    age = st.slider("Age", 0, 100, 50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking_status = st.selectbox("Smoking Status", ["Former", "Current", "Never"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    air_pollution = st.slider("Air Pollution Level", 0, 300, 100)
    biomass_exposure = st.selectbox("Biomass Fuel Exposure", [0, 1])
    occupational_exposure = st.selectbox("Occupational Exposure", [0, 1])
    family_history = st.selectbox("Family History of COPD", [0, 1])
    infections = st.selectbox("Respiratory Infections in Childhood", [0, 1])

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
    gender_male = 1 if gender == "Male" else 0
    smoking_former = 1 if smoking_status == "Former" else 0
    smoking_never = 1 if smoking_status == "Never" else 0
    smoking_current = 1 if smoking_status == "Current" else 0

    # Age bins (assuming 3 bins for simplicity)
    age_bin_adult = 1 if age < 40 else 0
    age_bin_middle_aged = 1 if 40 <= age < 60 else 0
    age_bin_elderly = 1 if age >= 60 else 0

    # Log-transformed features (using simple log if applicable)
    air_pollution_log = np.log(air_pollution + 1)  # Prevent log(0)
    bmi_log = np.log(bmi)  # Assuming bmi won't be zero

    # Create input array
    input_data = np.array([[age, biomass_exposure, occupational_exposure, family_history, 
                            bmi, air_pollution, infections, gender_male, smoking_former, 
                            smoking_never, smoking_current, *location_features,
                            air_pollution_log, bmi_log,
                            age_bin_adult, age_bin_middle_aged, age_bin_elderly]])
    
    return input_data

# Main app
st.title("COPD Prediction")
st.write("Enter the patient's information to predict the likelihood of COPD.")

# Get user input
input_data = get_user_input()

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write("COPD Likely")
    else:
        st.write("COPD Not Likely")