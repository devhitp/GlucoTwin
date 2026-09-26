import os
from pathlib import Path

# Provide a way to override dataset path with an environment variable.
# Fallback to local data/raw directory.
GLUCOTWIN_DATA_PATH = os.environ.get(
    "GLUCOTWIN_DATA_PATH", 
    str(Path(__file__).parent.parent.parent / "data" / "raw")
)
