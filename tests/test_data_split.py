from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_data
from src.data_cleaner import clean_data
from src.data_splitting import split_data

try:
    df = load_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    print("=== Data Split Successfully ===")
    print("Train Data Size", len(X_train))
    print("Test Data Size", len(X_test))
except Exception as e:
    print("Error Splitting the data", e)