import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import load_config

class DataLoadingError(Exception):
    pass

def load_data(path = None) -> pd.DataFrame:
    config = load_config()

    if path is None:
        project_root = Path(__file__).resolve().parent.parent
        path = project_root/Path(config.data.raw)
    else:
        path = Path(path)

    if not path.exists():
        raise DataLoadingError(f"Data not found in {path}")
    
    df = pd.read_csv(path)
    
    required_columns = config.features

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise DataLoadingError(f"Missing column(s): {missing_columns} in dataframe")
    
    return df
    