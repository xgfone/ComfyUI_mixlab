"""Shared API-key lookup for BIMO ComfyUI nodes.

Keys in ``<ComfyUI root>/conf/keys.json`` take precedence over environment
variables. Node execution never creates or alters the key file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def keys_file_path() -> Path:
    """Return the canonical ComfyUI key-file location."""
    try:
        import folder_paths

        base_path = getattr(folder_paths, "base_path", None)
        if base_path:
            return Path(base_path) / "conf" / "keys.json"
    except ImportError:
        pass

    # py/util/key_resolver.py -> py -> plugin -> custom_nodes -> ComfyUI root
    return Path(__file__).resolve().parents[4] / "conf" / "keys.json"


def _key_names(names: tuple[str, ...]) -> tuple[str, ...]:
    cleaned = tuple(name.strip() for name in names if isinstance(name, str) and name.strip())
    if not cleaned:
        raise ValueError("At least one key name is required.")
    return cleaned


def _file_keys() -> dict[str, str]:
    path = keys_file_path()
    if not path.is_file():
        return {}

    try:
        with path.open("r", encoding="utf-8-sig") as key_file:
            values = json.load(key_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in API key file: {path}") from error
    except OSError as error:
        raise ValueError(f"Unable to read API key file: {path}") from error

    if not isinstance(values, dict):
        raise ValueError(f"API key file must contain a JSON object: {path}")

    return {
        str(name): value.strip()
        for name, value in values.items()
        if isinstance(value, str) and value.strip()
    }


def get_key(*names: str, required: bool = True) -> str | None:
    """Get the first configured key, preferring ``conf/keys.json`` to env."""
    names = _key_names(names)
    file_keys = _file_keys()

    for name in names:
        if value := file_keys.get(name):
            return value
    for name in names:
        if value := os.getenv(name, "").strip():
            return value

    if required:
        raise ValueError(
            "API key is not configured. Set one of "
            f"{', '.join(names)} in {keys_file_path()} or in the environment."
        )
    return None


def get_key_name(*names: str) -> str | None:
    """Return the configured key's name without exposing its value."""
    names = _key_names(names)
    file_keys = _file_keys()

    for name in names:
        if file_keys.get(name):
            return name
    for name in names:
        if os.getenv(name, "").strip():
            return name
    return None
