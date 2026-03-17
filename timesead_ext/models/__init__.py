from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = ["other"]


def __getattr__(name: str) -> ModuleType:
    if name == "other":
        return import_module(".other", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
