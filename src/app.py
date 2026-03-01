from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import os
import random

app = Flask(__name__)
CORS(app)

# Load the model
# We wrap it in a try-except block to give a helpful error if the file is missing
model_path = '../models/fraud_model_final.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"✅ Model loaded successfully from {model_path}")
else:
    print(f"❌ Error: {model_path} not found. Please place the file in this directory.")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/simulate', methods=['GET'])
def simulate():
    """Generates a random transaction profile (either Safe or Fraud-like)"""
    # 50% chance to simulate a fraud attack, 50% chance for a normal user
    scenario = random.choice(['attack', 'normal'])
    
    if scenario == 'attack':
        # Simulate "Card Testing" or "Burst Attack"
        # High velocity (10-25) is the key marker for fraud here
        return jsonify({
            'amount': round(random.uniform(50, 1000), 2),
            'velocity': random.randint(10, 25), 
            'type': 'Simulated Attack'
        })
    else:
        # Simulate Normal Shopping
        # Low velocity (1-3) is typical for safe users
        return jsonify({
            'amount': round(random.uniform(10, 300), 2),
            'velocity': random.randint(1, 3), 
            'type': 'Normal Transaction'
        })

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        
        # Extract data from the frontend
        amount = float(data.get('amount'))
        velocity = float(data.get('velocity'))

        # Create DataFrame exactly like the training data
        # The column names MUST match what the model was trained on
        input_data = pd.DataFrame({
            'amount': [amount],
            'velocity_15m': [velocity]
        })

        # Get prediction probability
        # [0] gets the first row, [1] gets the probability of class "1" (Fraud)
        probability = model.predict_proba(input_data)[0][1]
        
        # Determine status (Threshold 0.5)
        is_fraud = probability > 0.5
        
        return jsonify({
            'probability_score': probability, # Raw float for the progress bar
            'probability_display': f"{probability:.1%}",
            'is_fraud': bool(is_fraud),
            'message': 'High Risk Transaction Detected' if is_fraud else 'Transaction Approved'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
