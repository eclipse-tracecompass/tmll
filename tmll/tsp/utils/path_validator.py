import os
from typing import Optional


class PathValidator:
    @staticmethod
    def is_path_valid(path: Optional[str]) -> bool:
        return path is not None and os.path.exists(path)
