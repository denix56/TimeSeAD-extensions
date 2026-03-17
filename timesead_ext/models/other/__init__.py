from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "NeutralAD",
    "NeutralADAnomalyDetector",
    "NeutralADLoss",
    "LNT",
    "LNTAnomalyDetector",
    "LNTLoss",
]

_SYMBOL_TO_MODULE = {
    "NeutralAD": ".neutral_ad",
    "NeutralADAnomalyDetector": ".neutral_ad",
    "NeutralADLoss": ".neutral_ad",
    "LNT": ".lnt",
    "LNTAnomalyDetector": ".lnt",
    "LNTLoss": ".lnt",
}


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name, __name__)
    return getattr(module, name)
