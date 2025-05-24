# src/training.py
from src import data_processing, feature_engineering, models, utils
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
import yaml

def load_config(config_path='src/config.yaml'):
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train():
    config = load_config()

    # Load data
    df = data_processing.get_data(config['raw_data_path'], config['processed_data_path'])

    # Encode labels
    df, label_encoder = feature_engineering.encode_labels(df, 'Label')

    # Select features
    X, y = feature_engineering.select_features(df, config['features'])

    # Scale features if desired
    X_scaled, scaler = feature_engineering.scale_features_in_chunks(X, chunk_size=100000)
    utils.save_scaler(scaler, config['scaler_path'])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train Random Forest
    rf = models.RFModel()
    rf.fit(X_train, y_train)
    rf.save(config['rf_model_path'])
    print("Random Forest trained and saved.")

    # Train Isolation Forest
    iso = models.IsolationForestModel()
    iso.fit(X_train)
    iso.save(config['iso_forest_model_path'])
    print("Isolation Forest trained and saved.")

    # --- New: Train SVM ---

    svm = OneClassSVM(kernel='rbf', nu=0.05)
    svm.fit(X_train)
    utils.save_model(svm, config['svm_model_path'])
    print("SVM trained and saved.")

if __name__ == "__main__":
    train()