from sklearn.metrics import f1_score, average_precision_score, confusion_matrix
import numpy as np

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "f1_score": f1_score(y_test, preds),
        "auc_pr": average_precision_score(y_test, probs),
        "confusion_matrix": confusion_matrix(y_test, preds)
    }
