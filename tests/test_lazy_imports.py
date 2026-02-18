import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_import_ordergrad_does_not_import_optional_backends():
    # Run in a fresh Python process so sys.modules is clean.
    code = r"""
import sys
import ordergrad  # noqa: F401

# Importing the top-level package should *not* import optional deps.
assert "torch" not in sys.modules, "torch should not be imported on ordergrad import"
assert "jax" not in sys.modules, "jax should not be imported on ordergrad import"
assert "jaxlib" not in sys.modules, "jaxlib should not be imported on ordergrad import"
"""
    p = _run(code)
    assert p.returncode == 0, f"subprocess failed:\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
