from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_data
from src.data_cleaner import clean_data
from src.data_splitting import split_data
from src.train_model import create_model, evaluate_model

try:
    df = load_data()
    df = clean_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    model = create_model(X_train, y_train, X_val, y_val)
    cm, ps, rs, f1, roc_auc, avg_ps, report = evaluate_model(model, X_test, y_test)
    print("Eval Metrics:", cm, ps, rs, f1, roc_auc, avg_ps, report)
    
    print("=== Data Split Successfully ===")
    print("Train Data Size", len(X_train))
    print("Test Data Size", len(X_test))
except Exception as e:
    print("Error Splitting the data", e)