"""orderstat_reward: expected order-statistics reward transforms.

This package provides three implementations of the same math:

- :mod:`orderstat_reward.numpy_backend` (always available)
- :mod:`orderstat_reward.torch_backend` (optional; requires PyTorch)
- :mod:`orderstat_reward.jax_backend` (optional; requires JAX + jaxlib)

Important: importing :mod:`orderstat_reward` **does not** import PyTorch/JAX.
Those libraries are imported only if/when you import the corresponding backend
module (or call :func:`get_backend` / :func:`get_transform` for that backend).
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any


__all__ = [
    "get_backend",
    "get_transform",
    "OrderStatTransform",
    "OrderStatKofN",
    "numpy_backend",
]


def get_backend(name: str) -> ModuleType:
    """Lazily import and return a backend module.

    Parameters
    ----------
    name:
        One of: ``"numpy"``, ``"np"``, ``"torch"``, ``"pytorch"``, ``"jax"``.

    Returns
    -------
    module
        The imported backend module.

    Raises
    ------
    ImportError
        If the requested backend's dependency is not installed.
    ValueError
        If `name` is not recognized.
    """

    key = str(name).lower().strip()
    if key in {"numpy", "np"}:
        return import_module("orderstat_reward.numpy_backend")
    if key in {"torch", "pytorch"}:
        try:
            return import_module("orderstat_reward.torch_backend")
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "PyTorch backend requested but could not be imported. "
                "Install optional deps with: `pip install orderstat-reward[torch]`"
            ) from e
    if key in {"jax"}:
        try:
            return import_module("orderstat_reward.jax_backend")
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "JAX backend requested but could not be imported. "
                "Install optional deps with: `pip install orderstat-reward[jax]`"
            ) from e

    raise ValueError(
        f"Unknown backend {name!r}. Expected one of: numpy/np, torch/pytorch, jax."
    )


def get_transform(name: str):
    """Convenience: return the backend's ``OrderStatTransform`` class."""
    return get_backend(name).OrderStatTransform


# Default alias: NumPy implementation.
OrderStatTransform = import_module("orderstat_reward.numpy_backend").OrderStatTransform
OrderStatKofN = OrderStatTransform



def __getattr__(name: str) -> Any:
    """Lazy attribute access for backend submodules.

    This allows:

    >>> import orderstat_reward as osr
    >>> np_backend = osr.numpy_backend

    and similarly for ``torch_backend`` / ``jax_backend``.
    """

    if name in {"numpy_backend", "torch_backend", "jax_backend"}:
        return import_module(f"orderstat_reward.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
