import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Function to load and preprocess new data
def preprocess_new_data(new_data, required_columns):
    new_data = pd.get_dummies(new_data)
    
    # Ensure the new data has the same columns as the training data
    for col in required_columns:
        if col not in new_data.columns:
            new_data[col] = 0
    
    new_data = new_data[required_columns]
    
    return new_data

def main():
    model_filepath = r"C:\Users\Rushik\OneDrive\Documents\Desktop\AI_Drug_Dosage_Optimization\models\dosage_model.pkl"
    scaler_filepath = r"C:\Users\Rushik\OneDrive\Documents\Desktop\AI_Drug_Dosage_Optimization\models\scaler.pkl"
    new_data_filepath = r"C:\Users\Rushik\OneDrive\Documents\Desktop\AI_Drug_Dosage_Optimization\data\new_patient_data.xlsx"
    output_filepath = r"C:\Users\Rushik\OneDrive\Documents\Desktop\AI_Drug_Dosage_Optimization\data\predicted_patient_data.xlsx"
    
    try:
        # Load the trained model and scaler
        model = joblib.load(model_filepath)
        scaler = joblib.load(scaler_filepath)
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return

    try:
        # Load and preprocess new data
        new_data = pd.read_excel(new_data_filepath)
        
        # Define the required columns as per the training data
        required_columns = ['patient_id', 'age', 'weight', 'genetic_marker2', 'current_medication_DrugA', 'current_medication_DrugB']
        
        new_data_processed = preprocess_new_data(new_data, required_columns)
        
        # Print the columns of the processed new data
        print("Columns in processed new data:")
        print(new_data_processed.columns)
        
        # Standardize features using the saved scaler
        new_data_scaled = scaler.transform(new_data_processed)
        
        # Make predictions
        predictions = model.predict(new_data_scaled)
        
        # Output predictions
        new_data['Predicted_Genetic_Marker1'] = predictions
        print(new_data)
        
        # Save predictions to a new file
        new_data.to_excel(output_filepath, index=False)
        print(f"Predictions saved to {output_filepath}")
        
    except Exception as e:
        print(f"Error processing data or making predictions: {e}")

if __name__ == "__main__":
    main()
