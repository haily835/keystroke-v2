import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report

def calculate_metrics(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Extract predictions and targets
    y_pred = df['pred']
    y_true = df['target']
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    # print(f"Recall: {recall:.4f}")


    # Return the metrics as a vector
    return f"{accuracy:.4f} & {precision:.4f} & {f1:.4f} & {recall:.4f}", y_pred, y_true
    # return [accuracy, precision, f1, recall]

