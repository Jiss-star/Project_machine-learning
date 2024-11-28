# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:47:10 2024

@author: jissm
"""

import numpy as np
import pickle
import streamlit as st

# Function to load the model
def load_model(model_path):
    """Load a trained model from a pickle file."""
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# Function to make predictions
def predict_wine_quality(classifier, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
                         free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol):
    """Predict wine quality based on features."""
    prediction = classifier.predict([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
                                     free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    return prediction[0]  # Extract the prediction result from the array

# Streamlit App
def main():
    st.title("Wine Quality Predictor")
    
    # HTML for styling
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Wine Quality Prediction App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Dropdown for model selection
    model_options = {
        "Model 1": r"C:\Users\jissm\1_JUPYTER_PROJECTS\Machine_learning\wine_streamlit\svc_wine.pkl",
        "Model 2": r"C:\Users\jissm\1_JUPYTER_PROJECTS\Machine_learning\wine_streamlit\gnb_wine.pkl",
        "Model 3": r"C:\Users\jissm\1_JUPYTER_PROJECTS\Machine_learning\wine_streamlit\knn_wine.pkl",
        "Model 4": r"C:\Users\jissm\1_JUPYTER_PROJECTS\Machine_learning\wine_streamlit\lr_wine.pkl",
        "Model 5": r"C:\Users\jissm\1_JUPYTER_PROJECTS\Machine_learning\wine_streamlit\rf_wine.pkl",
    }
    
    model_choice = st.selectbox("Select Model", list(model_options.keys()))
    selected_model_path = model_options[model_choice]
    
    # Load the selected model
    classifier = load_model(selected_model_path)
    
    # Input fields for features
    fixed_acidity = st.text_input("Fixed Acidity", "Type Here")
    volatile_acidity = st.text_input("Volatile Acidity", "Type Here")
    citric_acid = st.text_input("Citric Acid", "Type Here")
    residual_sugar = st.text_input("Residual Sugar", "Type Here")
    chlorides = st.text_input("Chlorides", "Type Here")
    free_sulfur_dioxide = st.text_input("Free Sulfur Dioxide", "Type Here")
    total_sulfur_dioxide = st.text_input("Total Sulfur Dioxide", "Type Here")
    density = st.text_input("Density", "Type Here")
    pH = st.text_input("pH", "Type Here")
    sulphates = st.text_input("Sulphates", "Type Here")
    alcohol = st.text_input("Alcohol", "Type Here")
    
    result = ""
    
    # Predict button
    if st.button("Predict"):
        try:
            # Convert inputs to float
            fixed_acidity = float(fixed_acidity)
            volatile_acidity = float(volatile_acidity)
            citric_acid = float(citric_acid)
            residual_sugar = float(residual_sugar)
            chlorides = float(chlorides)
            free_sulfur_dioxide = float(free_sulfur_dioxide)
            total_sulfur_dioxide = float(total_sulfur_dioxide)
            density = float(density)
            pH = float(pH)
            sulphates = float(sulphates)
            alcohol = float(alcohol)
            
            # Make prediction
            result = predict_wine_quality(classifier, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
                                         free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol)
            st.success(f'The predicted wine quality is: {result}')
        except ValueError:
            st.error("Please enter valid numerical inputs.")
    
    # About button
    if st.button("About"):
        st.text("This app predicts wine quality based on various chemical features.")
        st.text("Built with Streamlit and Python.")

if __name__ == '__main__':
    main()
