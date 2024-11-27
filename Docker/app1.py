# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:21:42 2024

@author: jissm
"""
import numpy as np
import pickle
import streamlit as st

# Load the trained model
pickle_in = open(r'C:\Users\jissm\1_JUPYTER_PROJECTS\Machine_learning\Docker\classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

# Function to make predictions
def predict_note_authentication(variance, skewness, curtosis, entropy):
    """Authenticate banknotes based on features."""
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return prediction[0]  # Extract the prediction result from the array

# Streamlit App
def main():
    st.title("Bank Authenticator")
    
    # HTML for styling
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Input fields for features
    variance = st.text_input("Variance", "Type Here")
    skewness = st.text_input("Skewness", "Type Here")
    curtosis = st.text_input("Curtosis", "Type Here")
    entropy = st.text_input("Entropy", "Type Here")
    
    result = ""
    
    # Predict button
    if st.button("Predict"):
        try:
            # Convert inputs to float
            variance = float(variance)
            skewness = float(skewness)
            curtosis = float(curtosis)
            entropy = float(entropy)
            
            # Make prediction
            result = predict_note_authentication(variance, skewness, curtosis, entropy)
            st.success(f'The output is: {result}')
        except ValueError:
            st.error("Please enter valid numerical inputs.")
    
    # About button
    if st.button("About"):
        st.text("This app authenticates banknotes using ML.")
        st.text("Built with Streamlit and Python.")

if __name__ == '__main__':
    main()

    
    
