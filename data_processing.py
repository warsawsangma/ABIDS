import pandas as pd
import os
import yaml
from src.utils import load_dataframe, save_dataframe, load_raw_data

def load_raw_data(raw_data_path):
    return pd.read_csv(raw_data_path)

def clean_data(df):
    # Basic cleaning, e.g., drop missing values
    df = df.dropna()
    return df

def process_and_save(raw_data_path, processed_data_path):
    df = load_raw_data(raw_data_path)
    df = clean_data(df)
    save_dataframe(df, processed_data_path)
    print(f"Processed data saved to {processed_data_path}")
    return df

def get_data(raw_data_path, processed_data_path):
    if os.path.exists(processed_data_path):
        print("Loading processed data...")
        df = load_dataframe(processed_data_path)
    else:
        print("Processing raw data...")
        df = process_and_save(raw_data_path, processed_data_path)
    return df