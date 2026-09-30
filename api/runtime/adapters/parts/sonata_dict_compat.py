"""Narrow adapter-local mapping semantics observed in pinned Sonata source.

This is not the upstream ``addict`` package. It supports the concrete
``Dict`` behavior used by Sonata: mapping construction, attribute/item access
and assignment, attribute deletion, and recursive conversion of nested plain
mappings in mappings/lists/tuples.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from types import ModuleType
from typing import Any


def _convert(value: Any) -> Any:
    if isinstance(value, SonataDictCompat):
        return value
    if isinstance(value, Mapping):
        return SonataDictCompat(value)
    if isinstance(value, list):
        return [_convert(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_convert(item) for item in value)
    return value


class SonataDictCompat(dict):
    """Small mapping with the attribute behavior exercised by Sonata's Point."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        if len(args) > 1:
            raise TypeError(f"expected at most 1 mapping argument, got {len(args)}")
        if args:
            source = args[0]
            if not isinstance(source, Mapping):
                source = dict(source)
            for key, value in source.items():
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, _convert(value))


def install_sonata_dict_compat() -> ModuleType:
    """Publish only the required import namespace for pinned Sonata imports."""
    existing = sys.modules.get("addict")
    if existing is not None:
        if getattr(existing, "_modly_sonata_dict_compat", False):
            return existing
        raise RuntimeError("addict is already imported; refusing to replace an existing module")
    module = ModuleType("addict")
    module.__file__ = __file__
    module.Dict = SonataDictCompat
    module._modly_sonata_dict_compat = True
    module.__doc__ = "Process-local Sonata Dict compatibility namespace; not upstream addict."
    sys.modules["addict"] = module
    return module
