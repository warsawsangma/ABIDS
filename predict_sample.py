# src/predict_sample.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src import feature_engineering, models, utils

def load_and_preprocess_sample(sample_df, feature_cols, scaler, label_encoder=None):
    """
    Process a new sample DataFrame:
    - select features
    - scale features
    - encode labels if label present
    """
    X_sample, _ = feature_engineering.select_features(sample_df, feature_cols)
    X_sample_scaled = scaler.transform(X_sample)
    return X_sample_scaled

def main():
    # Paths and configs
    model_path = 'models/rf_model.pkl'  # Update as needed
    model_path ='models/isolation_forest.pkl'
    model_path ='models/one_class_svm.pkl'
    scaler_path = 'models/scaler.pkl'   # Path where scaler was saved

    # Load scaler
    scaler = utils.load_scaler(scaler_path)
    X_sample = feature_engineering.select_features(sample_df, feature_cols)[0]
    X_sample_scaled = scaler.transform(X_sample)
    # Load the trained model
    rf_model = models.RFModel()
    rf_model.load(model_path)
    

    # Example: Load your sample data
    # Replace 'sample.csv' with your actual sample file
    sample_df = pd.read_csv('sample.csv')

    # Assuming your sample has the same feature columns as training
    feature_cols = ['feature1', 'feature2', 'feature3']  # replace with your actual features

    # If sample includes label and you want to encode it:
    # label_encoder = ... (load if needed)
    # X_sample = feature_engineering.select_features(sample_df, feature_cols)
    # If labels are present, encode them similarly

    # Process sample
    X_sample = load_and_preprocess_sample(sample_df, feature_cols, scaler)

    # Predict
    prediction = rf_model.predict(X_sample)
    print(f'Prediction: {prediction}')

if __name__ == "__main__":
    main()