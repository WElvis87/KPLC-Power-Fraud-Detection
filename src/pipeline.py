from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_data
from src.data_cleaner import clean_data
from src.data_splitting import split_data
from src.train_model import create_model, train_model, evaluate_model, create_evaluation_report, save_model

def run_pipeline():
    print("=== Starting Climalaria Pipeline ===")

    df = load_data()
    print("Data loaded:", df.shape)

    df = clean_data(df)
    print("Data cleaned:", df.shape)

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Data split: Train={len(X_train)}, Test={len(X_test)}")

    model = create_model()
    print("=== XGB Model Created Successfully ===")

    model = train_model(model, X_train, X_test, y_train, y_test)
    print("=== Model Trained Successfully ===")

    cm, ps, rs, f1, roc_auc, pr_auc, report, fraud_rate = evaluate_model(model, X_test, y_test)

    create_evaluation_report(cm, ps, rs, f1, roc_auc, pr_auc, report, fraud_rate)
    print("=== Evaluation Report Saved successfully ===")


    path = save_model(model)
    print(f"Model saved to {path}")

    print("=== Pipeline completed ===")

    return model

if __name__ == "__main__":
    run_pipeline()
    