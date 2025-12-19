from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_data
from src.data_cleaner import clean_data

try:
    df = load_data()
    df = clean_data(df)

    print("=== Data Cleaned Successfully ===")
    print("Columns:", df.columns.tolist())

except Exception as e:
    print("Failed to clean data", e)