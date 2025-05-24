    # src/utils.py
import numpy as np
import joblib
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler

def load_raw_data(path):
    """Load raw CSV data."""
    return pd.read_csv(path)

def save_model(model, path):
    directory = os.path.dirname(path)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)
    joblib.dump(model, path)

def load_model(model_path):
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = joblib.load(model_path)
        return model
    else:
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
def save_dataframe(df, path):
    """
    Save a pandas DataFrame to a file.
    
    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): Path to save the file.
    """
    # Ensure directory exists
    dir_name = os.path.dirname(path)
    if dir_name != '' and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    df.to_pickle(path)

def load_dataframe(path):
    """
    Load a pandas DataFrame from a file.
    
    Args:
        path (str): Path to the pickle file.
        
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"DataFrame file not found at: {path}")
    df = pd.read_pickle(path)
    return df

def save_test_data(X, y,path ):
    """
    its name dataframe because i use the function name in other code files but its actually test_data
    """
    data = {'X_test': X, 'y_test': y}
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_test_data(path):
    """
    its name dataframe because i use the function name in other code files but its actually test_data
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['X_test'], data['y_test']

def save_scaler(scaler, path):
    import joblib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)

def load_scaler(path):
    import joblib
    return joblib.load(path)

def save_pca(pca, path):
    import joblib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pca, path)

def load_pca(path):
    import joblib
    return joblib.load(path)

def print_separator():
    print("\n" + "-"*50 + "\n")


def process_in_chunks(data, chunk_size=100000):
    """
    Processes large DataFrame or numpy array in chunks:
    - Converts each chunk to float32
    - Cleans invalid values (NaNs, infinities)
    - Fills NaNs with zeros
    - Returns chunks as pandas DataFrames (if input was DataFrame) or numpy arrays
    """
    # Check if input is pandas DataFrame
    if isinstance(data, pd.DataFrame):
        total_rows = data.shape[0]
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            print(f"Processing rows {start} to {end}")
            chunk = data.iloc[start:end]
            # Convert to float32 numpy array
            chunk_array = chunk.to_numpy(dtype=np.float32)
            # Clean invalid values
            chunk_array = clean_invalid_values(chunk_array)
            # Convert back to DataFrame
            cleaned_chunk = pd.DataFrame(chunk_array, columns=data.columns)
            # Fill NaNs just in case
            cleaned_chunk = cleaned_chunk.fillna(0)
            yield cleaned_chunk

    elif isinstance(data, np.ndarray):
        total_rows = data.shape[0]
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            print(f"Processing rows {start} to {end}")
            chunk_array = data[start:end]
            # Convert to float32
            chunk_array = chunk_array.astype(np.float32)
            # Clean invalid values
            chunk_array = clean_invalid_values(chunk_array)
            yield chunk_array

    else:
        raise TypeError("Input data must be a pandas DataFrame or a numpy ndarray.")

def clean_invalid_values(array):
    """
    Replace infinities and NaNs with zeros.
    """
    # Replace inf and -inf with 0
    array = np.where(np.isfinite(array), array, 0)
    # Replace NaNs with 0
    array = np.nan_to_num(array, nan=0)
    return array