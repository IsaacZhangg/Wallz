"""Bundle all durable Colab campaign output for immediate local download."""

from __future__ import annotations

import tarfile
from pathlib import Path

OUTPUT = Path("/content/wallzero-output")
BUNDLE = Path("/content/wallzero-output.tar.gz")

if not OUTPUT.is_dir():
    raise FileNotFoundError(OUTPUT)
with tarfile.open(BUNDLE, "w:gz") as archive:
    archive.add(OUTPUT, arcname=OUTPUT.name)
print(BUNDLE, flush=True)
