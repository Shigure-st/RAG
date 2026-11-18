from pathlib import Path
from typing import Iterator


def get_file_path(dir_path: Path, extension: set[str]) -> Iterator[Path]:
    for file_path in dir_path.rglob('*'):
        if "/tests/" in str(file_path):
            continue
        elif "/examples/" in str(file_path):
            continue
        if file_path.is_file() and file_path.suffix in extension:
            yield file_path
