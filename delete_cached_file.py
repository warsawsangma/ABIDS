import os
import yaml

def load_config(config_path='src/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load your configuration
config = load_config()

# Paths to cache files
processed_data_path = config['processed_data_path']
scaler_path = config['scaler_path']
pca_path = config['pca_path']

cache_files = [processed_data_path, scaler_path, pca_path]

for file_path in cache_files:
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Deleted cache file: {file_path}")
        except Exception as e:
            print(f"Could not delete {file_path}: {e}")
    else:
        print(f"Cache file not found, skipping: {file_path}")