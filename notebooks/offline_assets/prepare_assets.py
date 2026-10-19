from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

models = list(Path("/kaggle/input").rglob("baseline.pt"))
assert len(models) == 1, "Attach exactly one completed training notebook output."
output = Path("/kaggle/working")
wheelhouse = output / "wheels"
wheelhouse.mkdir(parents=True, exist_ok=True)
shutil.copy2(models[0], output / "baseline.pt")
subprocess.check_call([
    sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--no-deps",
    "--dest", str(wheelhouse), "pydicom==3.0.2", "python-gdcm==3.2.6",
])
assert len(list(wheelhouse.glob("*.whl"))) == 2
manifest = {
    "python": list(sys.version_info[:2]),
    "checkpoint_sha256": hashlib.sha256((output / "baseline.pt").read_bytes()).hexdigest(),
    "wheels": [p.name for p in sorted(wheelhouse.glob("*.whl"))],
}
(output / "assets.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
