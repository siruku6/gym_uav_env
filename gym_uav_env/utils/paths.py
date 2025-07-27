import os
from pathlib import Path

APP_ROOT_DIR: Path = Path(os.path.abspath(__file__)).parents[2]

DATA_PATH: Path = APP_ROOT_DIR / "data"

OUTPUT_PATH: Path = APP_ROOT_DIR / "outputs"
