import yaml
import glob
import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Import your modules
from src import data_processing, feature_engineering, utils
from src.models.rf_model import RFModel
from src.models.isolation_forest_model import IsolationForestModel
from src.evaluation import evaluate_model  # Assuming this exists

from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

def load_config(config_path='src/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Anomaly Detection System')
    parser.add_argument('--model_type', type=str, choices=['rf', 'isolation_forest', 'one_class_svm'], required=True, help='Type of model to use')
    parser.add_argument('--mode', type=str, choices=['train', 'detect'], required=True, help='Operation mode: train or detect')
    args = parser.parse_args()

    config = load_config()

    # --- CSV combining ---
    csv_folder_path = r'D:\ABIDS-2\data\raw'
    combined_csv_path = os.path.join(csv_folder_path, 'network_traffic_data.csv')
    combined_pickle_path = os.path.join(csv_folder_path, 'combined_data.pkl')

    if os.path.exists(combined_pickle_path):
        print("Loading combined data from pickle...")
        df = pd.read_pickle(combined_pickle_path)
    else:
        print("Combining CSV files...")
        file_pattern = os.path.join(csv_folder_path, '*.csv')
        csv_files = glob.glob(file_pattern)
        csv_files = [f for f in csv_files if os.path.abspath(f) != os.path.abspath(combined_csv_path)]
        df_list = []
        for file in csv_files:
            if os.path.getsize(file) > 0:
                try:
                    df_temp = pd.read_csv(file)
                    df_list.append(df_temp)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
        if df_list:
            combined_df = pd.concat(df_list, ignore_index=True)
            combined_df.to_csv(combined_csv_path, index=False)
            combined_df.to_pickle(combined_pickle_path)
            print(f"Saved combined CSV and pickle.")
            df = combined_df
        else:
            print("No data found.")
            return

    # --- Encode Labels ---
    df, label_encoder = feature_engineering.encode_labels(df, 'Label')
    # --- Select Features ---
    X, y = feature_engineering.select_features(df, config['features'])

    # --- Clean data in chunks ---
    print("Cleaning data for scaling...")
    X_df = pd.DataFrame(X)

    def clean_numeric_strings(series):
        s_str = series.astype(str).replace('nan', '')
        return pd.to_numeric(s_str.str.extract(r'([\d.]+)')[0], errors='coerce')

    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            print(f"Cleaning column '{col}'...")
            X_df[col] = clean_numeric_strings(X_df[col])

    X_df.fillna(0, inplace=True)

    # Process in chunks
    all_chunks = []
    for chunk in utils.process_in_chunks(X_df):
        if isinstance(chunk, pd.DataFrame):
            all_chunks.append(chunk.to_numpy())
        elif isinstance(chunk, np.ndarray):
            all_chunks.append(chunk)
    X_cleaned = np.vstack(all_chunks)

    print(f"Features shape before PCA: {X_cleaned.shape}")
    print(f"Labels shape: {y.shape}")

    # --- PCA and scaling cache ---
    processed_data_path = config['processed_data_path']
    scaler_path = config['scaler_path']
    pca_path = config['pca_path']

    if os.path.exists(processed_data_path) and os.path.exists(scaler_path) and os.path.exists(pca_path):
        print("Loading cached PCA, scaler, and data...")
        X_pca = np.load(processed_data_path, allow_pickle=True)
        scaler = utils.load_scaler(scaler_path)
        pca = utils.load_pca(pca_path)
    else:
        print("Scaling features...")
        X_scaled, scaler = feature_engineering.scale_features_in_chunks(X_cleaned, chunk_size=100000)

        # PCA explained variance
        print("Calculating explained variance for PCA...")
        pca_full = PCA(n_components=X_scaled.shape[1])
        pca_full.fit(X_scaled)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)

        plt.figure(figsize=(8,5))
        plt.plot(range(1, X_scaled.shape[1]+1), cumvar, marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Variance Explained')
        plt.grid()
        plt.show()

        # Decide number of components
        desired_var = 0.95
        n_components = np.argmax(cumvar >= desired_var) + 1
        print(f"Retaining {n_components} PCA components to explain {desired_var*100}% variance.")

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        print(f"Shape after PCA: {X_pca.shape}")

        # Save cache
        utils.save_scaler(scaler, scaler_path)
        utils.save_pca(pca, pca_path)
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        np.save(processed_data_path, X_pca)

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # --- Clean test data ---
    print("Cleaning test data in chunks...")
    test_chunks = []
    for chunk in utils.process_in_chunks(pd.DataFrame(X_test)):
        if isinstance(chunk, pd.DataFrame):
            test_chunks.append(chunk.to_numpy())
        elif isinstance(chunk, np.ndarray):
            test_chunks.append(chunk)
    X_test_cleaned = np.vstack(test_chunks)

    # Save test data
    utils.save_test_data(X_test_cleaned, y_test, config['test_data_path'])

    if args.mode == 'train':
        # --- Train ---
        if args.model_type == 'rf':
            print("Training Random Forest...")
            rf = RFModel()
            rf.fit(X_train, y_train)
            utils.save_model(rf, config['rf_model_path'])
            print(f"Saved RF model at {config['rf_model_path']}")

        elif args.model_type == 'isolation_forest':
            print("Training Isolation Forest...")
            iso = IsolationForestModel()
            iso.fit(X_train)
            utils.save_model(iso, config['iso_forest_model_path'])
            print(f"Saved Isolation Forest at {config['iso_forest_model_path']}")

        elif args.model_type == 'one_class_svm':
            print("Training One-Class SVM...")
            svm = OneClassSVM(kernel='rbf', nu=0.05)
            svm.fit(X_train)
            utils.save_model(svm, config['svm_model_path'])
            print(f"Saved SVM at {config['svm_model_path']}")

    elif args.mode == 'detect':
        # --- Detection ---
        if args.model_type == 'rf':
            rf = utils.load_model(config['rf_model_path'])
            X_test_data, y_test = utils.load_test_data(config['test_data_path'])
            preds = rf.predict(X_test_data)
            probas = rf.predict_proba(X_test_data)

            threshold = 0.8
            for i, (pred, prob) in enumerate(zip(preds, probas)):
                prob_attack = prob[1]
                if pred == 1 or prob_attack > threshold:
                    print(f"Sample {i}: Anomaly (Pred: {pred}, Prob: {prob_attack:.2f})")
                else:
                    print(f"Sample {i}: Normal (Pred: {pred}, Prob: {prob_attack:.2f})")

            # Evaluate
            evaluation_results = evaluate_model(y_test, preds, 'Random Forest')
            print(evaluation_results)

        elif args.model_type == 'isolation_forest':
            iso = utils.load_model(config['iso_forest_model_path'])
            X_test_data, y_test = utils.load_test_data(config['test_data_path'])
            preds, scores = iso.predict_preds_and_scores(X_test_data)
            print(f"Predictions: {preds}")
            print(f"Scores: {scores}")

            # For evaluation, convert preds to labels if needed
            # Assuming preds are -1 (anomaly) or 1 (normal), convert to 0/1
            y_pred_labels = np.where(preds == -1, 1, 0)  # or adjust as per your labels
            y_true_labels = y_test  # assuming y_test is in same format

            evaluation_results = evaluate_model(y_true_labels, y_pred_labels, 'Isolation Forest')
            print(evaluation_results)

        elif args.model_type == 'one_class_svm':
            svm = utils.load_model(config['svm_model_path'])
            X_test_data, y_test = utils.load_test_data(config['test_data_path'])
            preds = svm.predict(X_test_data)
            scores = svm.decision_function(X_test_data)
            print(f"Predictions: {preds}")
            print(f"Scores: {scores}")

            # Convert preds if necessary
            y_pred_labels = np.where(preds == -1, 1, 0)
            y_true_labels = y_test

            evaluation_results = evaluate_model(y_true_labels, y_pred_labels, 'One-Class SVM')
            print(evaluation_results)

if __name__ == "__main__":
    main()