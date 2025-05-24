# src/models/one_class_svm_model.py
from sklearn.svm import OneClassSVM
import joblib
from src.utils import process_in_chunks, clean_invalid_values  # Import clean_invalid_values
import numpy as np
import pandas as pd

class SVMModel:
    def __init__(self, kernel='rbf', nu=0.05):
        self.model = OneClassSVM(kernel=kernel, nu=nu)
    
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
    
    def save(self, path):
        joblib.dump(self.model, path)
    
    def load(self, path):
        self.model = joblib.load(path)