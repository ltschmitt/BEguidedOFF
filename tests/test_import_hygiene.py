"""Import-hygiene guardrail: importing the fast stack must not pull torch/numba."""
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)

_PROBE = """
import sys
import beguidedoff
import beguidedoff.fast
import beguidedoff.fast.finder
import beguidedoff.fast.fmindex
banned = [m for m in ("torch", "numba") if m in sys.modules]
assert not banned, f"unexpected heavy imports: {banned}"
print("OK")
"""


def test_fast_imports_do_not_pull_torch_or_numba():
    env = dict(os.environ)
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
