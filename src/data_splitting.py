from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import load_config

class DataSplittingError(Exception):
    pass

def split_data(df):
    config = load_config()

    target = config.target
    features = config.features

    if target not in df.columns:
        raise DataSplittingError(f"Target Column Missing from dataframe")
    
    missing = [col for col in features if col not in df.columns]

    if missing: 
        raise DataSplittingError(f"Missing columns from your dataframe {missing}")

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test




    