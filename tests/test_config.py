from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import load_config

try:
    config = load_config()
    print("===Config file loaded Successfully===")
    print("Project Title: ", config.project.title)

except Exception as e:
    print(f"Failed to load Config file", e)