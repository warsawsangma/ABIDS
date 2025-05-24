 # src/models/isolation_forest_model.py
from sklearn.ensemble import IsolationForest
import joblib
from src.utils import process_in_chunks, clean_invalid_values  # Import process_in_chunks
import numpy as np
import pandas as pd

class IsolationForestModel:
    def __init__(self, n_estimators=100, contamination=0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)
    
    def fit(self, X):
        for X_chunk in process_in_chunks(X):
            X_array = X_chunk.to_numpy() if isinstance(X_chunk, pd.DataFrame) else X_chunk
            X_clean = clean_invalid_values(X_array)
            if X_clean.shape[0] > 0:
                self.model.fit(X_clean)
    
    def predict(self, X):
        predictions = []
        for X_chunk in process_in_chunks(X):
            X_array = X_chunk.to_numpy() if isinstance(X_chunk, pd.DataFrame) else X_chunk
            X_clean = clean_invalid_values(X_array)
            if X_clean.shape[0] > 0:
                predictions.extend(self.model.predict(X_clean))
            else:
                predictions.extend([-1] * (X_chunk.shape[0] if isinstance(X_chunk, pd.DataFrame) else X_chunk.shape[0]))
        return predictions
    
    def predict_scores(self, X):
        scores = []
        for X_chunk in process_in_chunks(X):
            X_array = X_chunk.to_numpy() if isinstance(X_chunk, pd.DataFrame) else X_chunk
            X_clean = clean_invalid_values(X_array)
            if X_clean.shape[0] > 0:
                scores.extend(self.model.decision_function(X_clean))
            else:
                scores.extend([0] * (X_chunk.shape[0] if isinstance(X_chunk, pd.DataFrame) else X_chunk.shape[0]))
        return scores

    def predict_preds_and_scores(self, X):
        preds = []
        scores = []
        for X_chunk in process_in_chunks(X):
            X_array = X_chunk.to_numpy() if isinstance(X_chunk, pd.DataFrame) else X_chunk
            X_clean = clean_invalid_values(X_array)
            n_samples = X_clean.shape[0]
            if n_samples > 0:
                preds.extend(self.model.predict(X_clean))
                scores.extend(self.model.decision_function(X_clean))
            else:
                preds.extend([-1] * (X_chunk.shape[0] if isinstance(X_chunk, pd.DataFrame) else X_chunk.shape[0]))
                scores.extend([0] * (X_chunk.shape[0] if isinstance(X_chunk, pd.DataFrame) else X_chunk.shape[0]))
        return preds, scores
    
    def save(self, path):
        joblib.dump(self.model, path)
    
    def load(self, path):
        self.model = joblib.load(path)