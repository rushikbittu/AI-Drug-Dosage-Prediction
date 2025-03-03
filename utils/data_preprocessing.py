import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath):
    # Load data
    data = pd.read_csv(filepath)
    
    # Feature selection and preprocessing
    features = data.drop(['patient_id', 'dosage_response'], axis=1)
    target = data['dosage_response']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data('data/patient_data.csv')
    print(X_train.shape, X_test.shape)
