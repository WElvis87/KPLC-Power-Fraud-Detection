from pathlib import Path
import yaml
from box import Box

class ErrorLoadingConfigFile(Exception):
    pass

def load_config(path:str = None) -> dict:
    if path is None:
        root_dir = Path(__file__).resolve().parent.parent
        path = root_dir/"config"/"config.yaml"
    else:
        path = Path(path)

    if not path.exists():
        raise ErrorLoadingConfigFile(f"Config File not found in {path}")
    
    with open (path, "r") as file:
        config = Box(yaml.safe_load(file))

    return config