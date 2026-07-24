"""Install an uploaded WallZero source archive into an active Colab runtime."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ARCHIVE = Path("/content/wallzero-source.tar.gz")
PROJECT = Path("/content/wallzero")


def main() -> None:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)
    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    PROJECT.mkdir(parents=True)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        archive.extractall(PROJECT, filter="data")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(PROJECT), "--no-deps"],
        check=True,
    )
    print(f"installed WallZero from {ARCHIVE} into {PROJECT}", flush=True)


main()
