from typing import Union
from pathlib import Path

from pytorch_lightning.utilities.rank_zero import rank_zero_only


def int_or_str(x: str)-> Union[str, int]:
    """ Function to cast to int or str. This is used to tackle precision 
    which can be int (16, 32) or str (bf16)"""
    try:
        return int(x)
    except ValueError:
        return x


@rank_zero_only
def unlink_file_or_folder(path: Path) -> None:
    """ Funtion to remove files and folders. Unlink works for files, rmdir
    for empty folders, but not for non-empty ones. Hence a recursive solution.
    """
    if path.exists():
        if path.is_file():
            path.unlink(missing_ok=True)
        else:
            for child in path.iterdir():
                unlink_file_or_folder(child)
            path.rmdir()