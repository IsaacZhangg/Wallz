"""Build a temporary self-extracting Colab script when file upload is blocked."""

from __future__ import annotations

import base64
import textwrap
from pathlib import Path

ARCHIVE = Path("/tmp/wallzero-source.tar.gz")
PAYLOAD = Path("/tmp/wallzero-colab-chunk.py")

encoded = base64.b64encode(ARCHIVE.read_bytes()).decode("ascii")
wrapped = "\n".join(textwrap.wrap(encoded, width=100))
PAYLOAD.write_text(
    f'''"""Generated WallZero source payload for an authorized Colab run."""
import base64
import shutil
import sys
import tarfile
import time
from pathlib import Path

DATA = """{wrapped}"""
archive_path = Path("/content/wallzero-source.tar.gz")
project = Path("/content/wallzero")
archive_path.write_bytes(base64.b64decode(DATA))
if project.exists():
    shutil.rmtree(project)
project.mkdir(parents=True)
with tarfile.open(archive_path, "r:gz") as archive:
    archive.extractall(project, filter="data")
sys.path.insert(0, str(project / "src"))

from wallzero.campaign import run_self_play_chunk
from wallzero.pipeline import preset

run_self_play_chunk(
    "/content/wallzero-output",
    preset("colab"),
    games=16,
    simulations=160,
    device_name="cuda",
)

output = Path("/content/wallzero-output")
bundle = Path("/content/wallzero-output.tar.gz")
with tarfile.open(bundle, "w:gz") as archive:
    archive.add(output, arcname=output.name)
print(f"ARTIFACT_READY:{{bundle}}", flush=True)
time.sleep(600)
''',
    encoding="utf-8",
)
print(PAYLOAD)
