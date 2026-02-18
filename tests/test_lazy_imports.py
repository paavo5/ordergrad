import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_import_orderstat_reward_does_not_import_optional_backends():
    # Run in a fresh Python process so sys.modules is clean.
    code = r"""
import sys
import orderstat_reward  # noqa: F401

# Importing the top-level package should *not* import optional deps.
assert "torch" not in sys.modules, "torch should not be imported on orderstat_reward import"
assert "jax" not in sys.modules, "jax should not be imported on orderstat_reward import"
assert "jaxlib" not in sys.modules, "jaxlib should not be imported on orderstat_reward import"
"""
    p = _run(code)
    assert p.returncode == 0, f"subprocess failed:\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
