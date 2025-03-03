from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the model and scaler
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'dosage_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')

model = joblib.load(model_path)
print("Model loaded successfully")

scaler = joblib.load(scaler_path)
print("Scaler loaded successfully")

def preprocess_new_data(new_data, required_columns):
    new_data = pd.get_dummies(new_data)
    
    # Ensure the new data has the same columns as the training data
    for col in required_columns:
        if col not in new_data.columns:
            new_data[col] = 0
    
    new_data = new_data[required_columns]
    
    return new_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        age = request.form.get('age', type=float)
        weight = request.form.get('weight', type=float)
        genetic_marker1 = request.form.get('genetic_marker1', type=float)
        genetic_marker2 = request.form.get('genetic_marker2', type=float)
        current_medication = request.form.get('current_medication')

        # Create a DataFrame from the form data
        new_data = pd.DataFrame([{
            'age': age,
            'weight': weight,
            'genetic_marker1': genetic_marker1,
            'genetic_marker2': genetic_marker2,
            'current_medication_DrugA': 1 if current_medication == 'DrugA' else 0,
            'current_medication_DrugB': 1 if current_medication == 'DrugB' else 0
        }])
        
        # Define the required columns as per the training data
        required_columns = ['age', 'weight', 'genetic_marker1', 'genetic_marker2', 'current_medication_DrugA', 'current_medication_DrugB']
        
        # Preprocess the new data
        new_data_processed = preprocess_new_data(new_data, required_columns)
        
        # Print processed data for debugging
        print("Processed Data:")
        print(new_data_processed)
        
        # Standardize features using the saved scaler
        new_data_scaled = scaler.transform(new_data_processed)
        
        # Print scaled data for debugging
        print("Scaled Data:")
        print(new_data_scaled)
        
        # Make predictions
        predictions = model.predict(new_data_scaled)
        
        # Output predictions
        dosage_class = predictions[0]
        
        # Map class label to dosage
        dosage_mapping = {0: '0 mg', 1: '50 mg', 2: '100 mg'}
        dosage = dosage_mapping[dosage_class]
        
        print(f"Predicted Dosage: {dosage}")
        
        return render_template('result.html', dosage=dosage)
        
    except Exception as e:
        print(f"Error: {str(e)}")  # Print error details
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
