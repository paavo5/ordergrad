"""ordergrad: expected order-statistics reward transforms."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

__all__ = ["get_backend", "get_transform", "OrderStatTransform", "numpy_backend"]


def get_backend(name: str) -> ModuleType:
    """Lazily import and return a backend module."""
    key = str(name).lower().strip()
    if key == "numpy":
        return import_module("ordergrad.numpy_backend")
    if key == "torch":
        try:
            return import_module("ordergrad.torch_backend")
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "PyTorch backend requested but could not be imported. "
                "Install optional deps with: `pip install ordergrad[torch]`"
            ) from e
    if key == "jax":
        try:
            return import_module("ordergrad.jax_backend")
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "JAX backend requested but could not be imported. "
                "Install optional deps with: `pip install ordergrad[jax]`"
            ) from e

    raise ValueError(f"Unknown backend {name!r}. Expected one of: numpy, torch, jax.")


def get_transform(name: str):
    return get_backend(name).OrderStatTransform


OrderStatTransform = import_module("ordergrad.numpy_backend").OrderStatTransform



def __getattr__(name: str) -> Any:
    if name in {"numpy_backend", "torch_backend", "jax_backend"}:
        return import_module(f"ordergrad.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
