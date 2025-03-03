import tensorflow as tf
import numpy as np

def load_model(model_path):
    """
    Load the trained model from the given path.

    Parameters:
    model_path (str): Path to the trained model.

    Returns:
    model: Loaded TensorFlow model.
    """
    model = tf.keras.models.load_model(model_path)
    return model

def predict_dosage(model, patient_data):
    """
    Predict the optimal drug dosage for a patient based on their data.

    Parameters:
    model: The trained TensorFlow model.
    patient_data (numpy array): Array containing patient data.

    Returns:
    float: Predicted optimal dosage.
    """
    patient_data = np.array([patient_data]).astype(float)
    predicted_dosage = model.predict(patient_data)
    return predicted_dosage[0][0]

if __name__ == "__main__":
    # Example usage
    model = load_model('../models/dosage_model.h5')
    sample_patient_data = [45, 70, 1, 0, 1, 2]  # Example patient data
    dosage = predict_dosage(model, sample_patient_data)
    print(f"Predicted Dosage: {dosage} mg")
