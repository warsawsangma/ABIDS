# src/feature_engineering.py
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def encode_labels(df, label_col='Label'):
    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col])
    return df, le

def select_features(df, feature_cols):
    """
    Select specified feature columns and return as NumPy arrays.
    """
    X = df[feature_cols]
    y = df['Label']
    return X.values, y.values  # Convert to NumPy arrays


def scale_features_in_chunks(X, chunk_size=100000):
    # Convert to float32 to reduce memory footprint
    X = X.astype(np.float32)
    
    scaler = StandardScaler()
    n_samples = X.shape[0]
    
    # Incrementally fit the scaler
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        scaler.partial_fit(X[start:end])
    
    # Incrementally transform the data
    X_scaled = np.empty_like(X)
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        X_scaled[start:end] = scaler.transform(X[start:end])
        
    return X_scaled, scaler