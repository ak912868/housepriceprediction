from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model and label encoders
model = pickle.load(open("model/house_price_model.pkl", "rb"))
label_encoders = pickle.load(open("model/label_encoders.pkl", "rb"))

# Define the expected feature order
categorical_cols = ['mainroad', 'guestroom', 'basement',
                    'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs
    input_data = {
        'area': float(request.form['area']),
        'bedrooms': int(request.form['bedrooms']),
        'bathrooms': int(request.form['bathrooms']),
        'stories': int(request.form['stories']),
        'parking': int(request.form['parking']),
        'mainroad': request.form['mainroad'],
        'guestroom': request.form['guestroom'],
        'basement': request.form['basement'],
        'hotwaterheating': request.form['hotwaterheating'],
        'airconditioning': request.form['airconditioning'],
        'prefarea': request.form['prefarea'],
        'furnishingstatus': request.form['furnishingstatus'],
    }

    # Encode categorical inputs
    for col in categorical_cols:
        le = label_encoders[col]
        input_data[col] = le.transform([input_data[col]])[0]

    # Create DataFrame in correct order
    input_df = pd.DataFrame([input_data], columns=numerical_cols + categorical_cols)

    # Predict
    prediction = model.predict(input_df)[0]
    return render_template('result.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
