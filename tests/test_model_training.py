from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_data
from src.data_cleaner import clean_data
from src.data_splitting import split_data
from src.train_model import create_model, train_model, evaluate_model

try:
    df = load_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model = create_model()
    model = train_model(model, X_train, X_test, y_train, y_test)
    cm, ps, rs, f1, roc_auc, pr_auc, report, fraud_rate = evaluate_model(model, X_test, y_test)
    print("=== Model created successfully ===")

    print("Confusion Matrix:", cm)
    print("Precision Score:", ps)
    print("Recall Score:", rs)
    print("F1 Score:", f1)
    print("ROC AUC SCORE:", roc_auc)
    print("Precision Score:", pr_auc)
    print("Classification Report:", report)
    print("Fraud Rate:", fraud_rate)
except Exception as e:
    print("Error Splitting the data", e)