# utils.evaluation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score,
    confusion_matrix
    )

def category_metrics(y_test : list, y_pred : list):
    category_names = sorted(pd.Series(y_test).unique())
    precision = precision_score(y_test, y_pred, average=None, labels=category_names, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, labels=category_names, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, labels=category_names, zero_division=0)
    return pd.DataFrame({
        "Category": category_names,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })



def create_results_df(X_test, y_test, y_pred):
    X_test = pd.Series(X_test, name="X_test")
    y_test = pd.Series(y_test, name="y_test")
    y_pred = pd.Series(y_pred, name="y_pred")

    if not (len(X_test) == len(y_test) == len(y_pred)):
        raise ValueError(f"Length mismatch: X_test({len(X_test)}), y_test({len(y_test)}), y_pred({len(y_pred)})")

    results = pd.DataFrame({
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    })
    results["correct"] = results["y_test"] == results["y_pred"]
    return results


def performance_breakdown(results : pd.DataFrame):
    correct = np.sum(results["correct"])
    unknown = np.sum(results["y_pred"] == 'unknown')
    errors = np.sum(results["y_pred"] == 'error')
    incorrect = len(results) - correct - errors
    
    performance = pd.DataFrame({
        'Outcome': ['Correct', 'Incorrect', 'Unknown', 'Error'],
        'Count': [correct,incorrect,unknown, errors]
        })
    return performance


def evaluation(y_test : list, y_pred : list):
    if not (len(y_test) == len(y_pred)):
        raise ValueError(f"Length mismatch: y_test({len(y_test)}), y_pred({len(y_pred)})")

    # Compute overall accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Get category labels (sorted for consistency)
    category_names = sorted(pd.Series(y_test).unique())

    # Compute per-class metrics
    precision = precision_score(y_test, y_pred, average=None, labels=category_names, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, labels=category_names, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, labels=category_names, zero_division=0)

    # Store per-category metrics
    metrics_df = pd.DataFrame({
        "Category": category_names,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    return accuracy, metrics_df



def plot_metrics(accuracy, metrics_df):
    '''
        metrics_df : df with following format 
            metrics_df = pd.DataFrame({
                "Category": labels,
                "Precision": class_precision,
                "Recall": class_recall,
                "F1 Score": class_f1
            })
    '''
    categories = metrics_df['Category']
    
    plt.figure(figsize=(8, 5))
    
    plt.plot(categories, metrics_df['F1 Score'], marker='o', label='F1 Score')
    plt.plot(categories, metrics_df['Precision'], marker='s', label='Precision')
    plt.plot(categories, metrics_df['Recall'], marker='^', label='Recall')
    
    plt.axhline(y=accuracy, color='r', linestyle='--', label=f'Accuracy ({accuracy:.2f})')
    
    plt.xlabel("Category")
    plt.ylabel("Score")
    plt.title("Evaluation Metrics per Category")
    plt.xticks(categories, categories, rotation=45, ha="right")
    plt.legend()
    plt.grid(True)

    print()
    plt.show()

    print("Category metrics")
    print(metrics_df.round(2))
    print()



def plot_confusion_matrix(y_test : list, y_pred : list):

    if not (len(y_test) == len(y_pred)):
        raise ValueError(f"Length mismatch: y_test({len(y_test)}), y_pred({len(y_pred)})")

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 3))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=np.unique(y_test), 
        yticklabels=np.unique(y_test)
        )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()