# src/models/rf_model.py
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
from src.utils import clean_invalid_values

class RFModel:
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.rf = None

    def fit(self, X, y):
        # Convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)

        # Clean invalid values
        X_clean = clean_invalid_values(X)
        # Fit the Random Forest classifier
        self.rf = RandomForestClassifier(n_estimators=self.n_estimators)
        self.rf.fit(X_clean, y)

    def save(self, path):
        # Save only the trained model
        joblib.dump(self.rf, path)

    def load(self, path):
        # Load trained model
        self.rf = joblib.load(path)

    def predict(self, X):
        X = np.asarray(X)
        predictions = []

        for start_idx in range(0, X.shape[0], 10000):
            end_idx = min(start_idx + 10000, X.shape[0])
            X_chunk = X[start_idx:end_idx]

            X_clean = clean_invalid_values(X_chunk)
            if X_clean.shape[0] > 0:
                preds = self.rf.predict(X_clean)
                predictions.extend(preds)
            else:
                # Handle empty chunks
                predictions.extend([0] * (end_idx - start_idx))
        return predictions

    def predict_proba(self, X):
        X = np.asarray(X)
        probas = []

        for start_idx in range(0, X.shape[0], 10000):
            end_idx = min(start_idx + 10000, X.shape[0])
            X_chunk = X[start_idx:end_idx]

            X_clean = clean_invalid_values(X_chunk)
            if X_clean.shape[0] > 0:
                prob = self.rf.predict_proba(X_clean)
                probas.extend(prob)
            else:
                # Handle empty chunks
                probas.extend([[0.5, 0.5]] * (end_idx - start_idx))
        return probas