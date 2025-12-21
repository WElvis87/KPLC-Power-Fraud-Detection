from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_data
from src.data_cleaner import clean_data
from src.data_splitting import split_data
from src.train_model import create_model, evaluate_model, save_model

def run_pipeline():
    print("=== Starting Climalaria Pipeline ===")

    df = load_data()
    print("Data loaded:", df.shape)

    df = clean_data(df)
    print("Data cleaned:", df.shape)

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Data split: Train={len(X_train)}, Test={len(X_test)}")

    model = create_model(model, X_train, y_train)
    print("Model trained.")

    metrics = evaluate_model(model, X_test, y_test)
    print(f"Evaluation metrics: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")

    path = save_model(model)
    print(f"Model saved to {path}")

    print("=== Pipeline completed ===")

    return model, metrics

if __name__ == "__main__":
    run_pipeline()