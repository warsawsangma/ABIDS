import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(y_true, y_pred, model_name):
    """Evaluates a model's performance using various metrics.

    Args:
        y_true: True labels (e.g., 1 for normal, -1 for anomaly).
        y_pred: Predicted labels (e.g., 1 for normal, -1 for anomaly).
        model_name: Name of the model (string, for output).

    Returns:
        A dictionary containing the evaluation metrics.
        Returns None if input arrays have different lengths.
    """

    if len(y_true) != len(y_pred):
        print("Error: Input arrays have different lengths.")
        return None

    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1)  # Adjust pos_label if needed
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    
    #Example for ROC AUC
    #roc_auc = roc_auc_score(y_true, y_pred) 
    
    print(f"\nEvaluation Results for {model_name}:")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    #print(f"ROC AUC: {roc_auc:.4f}")  # Uncomment if you want to include ROC AUC

    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        # 'roc_auc': roc_auc,
        'confusion_matrix': cm
    }


