# Import the Libraries
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import os

# Load the trained model

try:
    model_path = os.path.join(os.path.dirname(__file__), 'Best_Random_Forest_Model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file not found. Please ensure the model file exists at the specified path.")
    st.stop()  # Stop execution if the model cannot be loaded
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()  # Stop execution for any other error

def get_user_input():
    Age = st.slider("Age", 0, 100, 50)
    Family_History_COPD = st.selectbox("Family History of COPD", [0, 1])
    BMI = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    Air_pollution_Level = st.slider("Air Pollution Level", 0, 300, 100)
    Pollution_Risk_Score = st.selectbox("Pollution Risk Score", [0, 1])
    Smoking_Status = st.selectbox("Smoking Status", ["Former", "Current", "Never"])
    Smoking_Pollution_interaction = st.number_input("Smoking Pollution Interaction", min_value=10.0, max_value=50.0, value=25.0)

    # Gender encoding
    gender_male = 1 if st.selectbox("Gender", ["Male", "Female"]) == "Male" else 0
    gender_female = 1 - gender_male  # This should be 1 if male is 0

    # Smoking status encoding
    smoking_former = 1 if Smoking_Status == "Former" else 0
    smoking_never = 1 if Smoking_Status == "Never" else 0
    smoking_current = 1 if Smoking_Status == "Current" else 0  # Include this for completeness

    # One-hot encoding for locations
    location = st.selectbox("Location", [
        "Bhaktapur", "Biratnagar", "Butwal", "Chitwan", 
        "Dharan", "Hetauda", "Kathmandu", "Lalitpur", 
        "Nepalgunj", "Pokhara"
    ])
    
    # Create one-hot encoded features for locations
    location_features = [1 if loc == location else 0 for loc in [
        "Bhaktapur", "Biratnagar", "Butwal", "Chitwan", 
        "Dharan", "Hetauda", "Kathmandu", "Lalitpur", 
        "Nepalgunj", "Pokhara"
    ]]

    # Create Age categories
    age_category_30_39 = 1 if 30 <= Age < 40 else 0
    age_category_40_49 = 1 if 40 <= Age < 50 else 0
    age_category_50_59 = 1 if 50 <= Age < 60 else 0
    age_category_60_69 = 1 if 60 <= Age < 70 else 0
    age_category_70_79 = 1 if 70 <= Age < 80 else 0

    # Create BMI categories
    bmi_category_normalweight = 1 if 18.5 <= BMI < 24.9 else 0
    bmi_category_obesity = 1 if BMI >= 30 else 0
    bmi_category_overweight = 1 if 25 <= BMI < 30 else 0

    # Create input array
    input_data = np.array([[Age, Family_History_COPD, BMI, Air_pollution_Level, 
                            Pollution_Risk_Score, Smoking_Pollution_interaction,
                            gender_female, gender_male,
                            smoking_former, smoking_never, smoking_current] +
                           location_features +
                           [age_category_30_39, age_category_40_49, age_category_50_59,
                            age_category_60_69, age_category_70_79,
                            bmi_category_normalweight, bmi_category_obesity, 
                            bmi_category_overweight]])

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
