from __future__ import annotations

import shutil


def resolve_binary(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(f"{name} was not found. Install it and add it to PATH.")
