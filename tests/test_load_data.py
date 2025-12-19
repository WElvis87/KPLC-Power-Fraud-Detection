from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.data_loader import load_data

try:
    config = load_config()
    df = load_data()

    print("=== Data Loaded Successfully ===")
    print("Columns:", df.columns.tolist())

except Exception as e:
    print("Failed to load data", e)