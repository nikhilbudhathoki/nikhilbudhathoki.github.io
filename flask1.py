from flask import Flask, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_KEY', 'default_secret_key')

# Load the model
model_path = 'train_model.sav'
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
else:
    raise FileNotFoundError(f"Model file not found: {model_path}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.json['input_data']
        
        # Process the data
        data.append(data[5] + data[6])  # Combine ApplicantIncome and CoapplicantIncome
        data.pop(5)  # Remove ApplicantIncome
        data.pop(5)  # Remove CoapplicantIncome
        
        # Convert list to numpy array and reshape for prediction
        input_data = np.asarray(data).reshape(1, -1)
        
        # Make prediction
        result = loaded_model.predict(input_data)
        
        # Interpret the result
        status = "✅ Loan Accepted" if result == 1 else "❌ Loan Not Accepted"
        
        return jsonify({'status': status})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0')
