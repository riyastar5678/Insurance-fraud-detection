from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load model and training columns
model_path = r'D:\Insurance\xgb_model.pkl'
columns_path = r'D:\Insurance\model_columns.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(columns_path, 'rb') as f:
    model_columns = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read form input
        form_data = request.form.to_dict()

        # Convert numeric values
        numeric_cols = ['MonthsAsCustomer', 'PolicyDeductable', 'UmbrellaLimit', 
                        'InjuryClaim', 'PropertyClaim', 'VehicleClaim', 'BodilyInjuries', 'Witnesses',
                        'CapitalGains', 'CapitalLoss', 'IncidentHourOfTheDay']
        for col in numeric_cols:
            if col in form_data:
                form_data[col] = float(form_data[col])
            else:
                form_data[col] = 0.0

        # Create DataFrame from form
        input_df = pd.DataFrame([form_data])

        # One-hot encode categorical variables to match training columns
        input_df = pd.get_dummies(input_df)

        # Add missing columns from training
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Ensure column order matches training
        input_df = input_df[model_columns]

        # Predict
        pred = model.predict(input_df)[0]

        return render_template('index.html', prediction_text=f"Fraud Prediction: {pred}")
    
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
