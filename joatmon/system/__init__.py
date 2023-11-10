import os
import shutil
from pathlib import Path


def is_installed(lib_name: str) -> bool:
    lib = shutil.which(lib_name)
    if lib is None:
        return False
    global_path = Path(lib)
    # else check if path is valid and has the correct access rights
    return global_path.exists() and os.access(global_path, os.X_OK)
