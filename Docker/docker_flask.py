# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
app=Flask(__name__)#Which point to start
#execution of the line cause the running of the app
pickle_in = open(r'C:\Users\jissm\1_JUPYTER_PROJECTS\Machine_learning\Docker\classifier.pkl', 'rb')

#pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)
#WSGI-WEB SERVER GATE INTERFACE
@app.route('/')# move to this app first trigger this functions
def welcome():
    return "Welcome all"
@app.route('/predict')#Decorater
def predict_note_autentication():#define variance name
## Retrieve and convert the input parameters
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The predicted values is"+str(prediction)
    
#http://127.0.0.1:5000/predict?variance=2&skewness=3&curtosis=2&entropy=1
@app.route('/predict_file', methods=["POST"])
def predict_note_file():
    """
    Authenticate the Bank Notes using a file.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The output values
    """
    try:
        # Get the uploaded file
        uploaded_file = request.files.get("file")
        if not uploaded_file:
            return "Error: No file uploaded. Please provide a valid CSV file.", 400

        # Read the file into a Pandas DataFrame
        df_test = pd.read_csv(uploaded_file)
        print(df_test.head())  # Debugging

        # Make predictions
        prediction = classifier.predict(df_test)
        return "The predicted values for CSV are: " + str(list(prediction))
    except Exception as e:
        return f"Error: {e}", 500


if __name__=='__main__':
    app.run()
    
    

