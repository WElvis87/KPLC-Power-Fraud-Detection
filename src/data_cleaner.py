import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import load_config

class DataCleaningError(Exception):
    pass

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    config = load_config()
    df = df.copy()

    columns = config.features + [config["target"]]

    missing_columns = [col for col in columns if col not in df.columns]

    if missing_columns:
        raise DataCleaningError(f"Missing Columns{missing_columns}")
    
    df = df.dropna(subset=columns)

    df = df.drop_duplicates()

    k = 1.5

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - k * IQR
        upper = Q3 + k * IQR

        df[col] = df[col].clip(lower, upper)

    mapping = {"low": 0, "medium": 1, "high": 2}
    df["household_type_id"] = df["household_type"].map(mapping)

    if df["household_type_id"].isna().any():
        raise DataCleaningError(f"Unexpected value in the householdtype")
    
    location_mapping = {"Eldoret" : 0, "Mombasa": 1, "Kisumu": 2, "Nakuru": 3, "Malindi": 4, "Nairobi": 5, "Nyeri": 6, "Thika": 7}
    df["region_id"] = df["region"].map(location_mapping)

    if df["region_id"].isna().any():
        raise DataCleaningError(f"Unexpected Location found")

    return df

