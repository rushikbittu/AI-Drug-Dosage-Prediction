import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

def map_dosage_to_classes(dosage_response):
    if dosage_response < 5:
        return 0  # 0 mg
    elif 5 <= dosage_response < 10:
        return 1  # 50 mg
    elif 10 <= dosage_response < 15:
        return 2  # 100 mg
    else:
        return 2  # Assuming 100 mg for values above 15

def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path)
    df = df.dropna()
    df['dosage_class'] = df['dosage_response'].apply(map_dosage_to_classes)
    df = pd.get_dummies(df)

    X = df.drop(['genetic_marker1', 'dosage_response', 'dosage_class'], axis=1)
    y = df['dosage_class']
    
    return X, y

def plot_correlation_matrix(X, y):
    # Combine features and target for correlation matrix plot
    df_combined = pd.concat([X, y], axis=1)
    plt.figure(figsize=(12, 8))
    correlation_matrix = df_combined.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.show()

def tune_hyperparameters(X_train, y_train):
    # Define parameter grid for RandomForest hyperparameter tuning
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    # Use GridSearchCV to find the best model based on the parameter grid
    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=3)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters found by GridSearchCV
    print("Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

def save_model(model, model_filepath, scaler, scaler_filepath):
    # Save the trained model and scaler objects to disk
    joblib.dump(model, model_filepath)
    joblib.dump(scaler, scaler_filepath)
    print(f"Model and scaler saved to {model_filepath} and {scaler_filepath}")

def main():
    # Define file paths for data, model, and scaler
    data_filepath = r"C:\Users\Rushik\OneDrive\Documents\Desktop\AI_Drug_Dosage_Optimization\data\patient_data.xlsx"
    model_filepath = r"C:\Users\Rushik\OneDrive\Documents\Desktop\AI_Drug_Dosage_Optimization\models\dosage_model.pkl"
    scaler_filepath = r"C:\Users\Rushik\OneDrive\Documents\Desktop\AI_Drug_Dosage_Optimization\models\scaler.pkl"

    # Load and preprocess the data
    X, y = load_and_preprocess_data(data_filepath)
    print("Class distribution in the target variable:")
    print(y.value_counts())

    # Plot the correlation matrix for feature inspection
    plot_correlation_matrix(X, y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Tune hyperparameters using GridSearchCV and train the best model
    model = tune_hyperparameters(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model performance using accuracy score, confusion matrix, and classification report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model and scaler to files
    save_model(model, model_filepath, scaler, scaler_filepath)

# Execute the main function when the script is run
if __name__ == "__main__":
    main()
