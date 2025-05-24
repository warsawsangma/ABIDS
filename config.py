import yaml

# Path to your YAML config file
CONFIG_FILE = 'config.yaml'

# Load the YAML config
with open(CONFIG_FILE, 'r') as file:
    config = yaml.safe_load(file)

# Assign variables
RAW_DATA_PATH = config.get('RAW_DATA_PATH')
PROCESSED_DATA_PATH = config.get('PROCESSED_DATA_PATH')