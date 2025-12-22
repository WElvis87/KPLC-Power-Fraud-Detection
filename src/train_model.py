import xgboost as xgb
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, average_precision_score, classification_report
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import load_config

class ErrorTrainingModel(Exception):
    pass

def create_model():
    config = load_config()
    params = config.model1.parameters

    model = xgb.XGBClassifier(**params)

    return model

def train_model(model, X_train, X_test, y_train, y_test):
    model = model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    ps = precision_score(y_test, y_pred)
    rs = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    fraud_rate = np.mean(y_test)

    return cm, ps, rs, f1, roc_auc, pr_auc, report, fraud_rate

def create_evaluation_report(cm, ps, rs, f1, roc_auc, pr_auc, report, fraud_rate, output_path=None):
    tn, fp, fn, tp = cm.ravel()
    config = load_config()
    if output_path is None:
        output_path = config.output.evaluation_report

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Model Performance Evaluation Report\n\n")

        f.write("## Confusion Matrix\n")
        f.write("| | Predicted Normal | Predicted Fraud |\n")
        f.write("|---|---|---|\n")
        f.write(f"| Actual Normal | {tn} | {fp} |\n")
        f.write(f"| Actual Fraud | {fn} | {tp} |\n\n")

        f.write("## Classification Metrics (thresholded at 0.5)\n")
        f.write(f"- Precision: {ps:.3f}\n")
        f.write(f"- Recall: {rs:.3f}\n")
        f.write(f"- F1-score: {f1:.3f}\n\n")

        f.write("## Threshold-independent Metrics\n")
        f.write(f"- ROC-AUC: {roc_auc:.3f}\n")
        f.write(f"- PR-AUC: {pr_auc:.3f} (Baseline ≈ {fraud_rate:.3f})\n\n")

        f.write("## Detailed Metrics from sklearn\n")
        f.write(f"{report}\n\n")

        f.write("## Interpretation\n")
        f.write(
            f"The model demonstrates strong ranking ability "
            f"(ROC-AUC={roc_auc:.3f}) and a trade-off between recall ({rs:.3f}) "
            f"and precision ({ps:.3f}). Missed fraud cases (FN={fn}) may result in revenue loss, "
            f"while false positives (FP={fp}) increase inspection costs.\n"
        )

    print(f"Report generated: {output_path}")

def save_model(model):
    config = load_config()
    path = config.output.model_path
    joblib.dump(model, path)
    path


