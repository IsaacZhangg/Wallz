"""Restore a previously downloaded/uploaded durable campaign bundle."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

BUNDLE = Path("/content/wallzero-output.tar.gz")
OUTPUT = Path("/content/wallzero-output")

if not BUNDLE.is_file():
    raise FileNotFoundError(BUNDLE)
if OUTPUT.exists():
    shutil.rmtree(OUTPUT)
with tarfile.open(BUNDLE, "r:gz") as archive:
    archive.extractall("/content", filter="data")
print(OUTPUT, flush=True)
