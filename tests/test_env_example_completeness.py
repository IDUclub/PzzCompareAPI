"""Every setting the code demands must be documented in .env.example.

A deployment builds its .env from the repository's Actions variables, and
.env.example is the checklist those variables are written against. Twelve
required keys were missing from it, so a fresh checkout could not start:
``iduconfig.Config.get`` raises on an absent key rather than defaulting.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / ".env.example"
SOURCE_ROOTS = ("service", "pipeline_modules")
REQUIRED_PATTERNS = (
    re.compile(r'config\.get\(\s*"([A-Z0-9_]+)"'),
    re.compile(r'_get_required_env\(\s*config,\s*"([A-Z0-9_]+)"'),
)


def documented_keys() -> set[str]:
    keys = set()
    for line in EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            keys.add(stripped.split("=", 1)[0].strip())
    return keys


def required_keys() -> dict[str, str]:
    found: dict[str, str] = {}
    for root in SOURCE_ROOTS:
        for path in (ROOT / root).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for pattern in REQUIRED_PATTERNS:
                for key in pattern.findall(text):
                    found.setdefault(key, str(path.relative_to(ROOT)))
    return found


def test_env_example_documents_every_required_key() -> None:
    documented = documented_keys()
    missing = {key: where for key, where in required_keys().items() if key not in documented}

    assert not missing, "keys read without a default but absent from .env.example: " + ", ".join(
        f"{key} ({where})" for key, where in sorted(missing.items())
    )


@pytest.mark.parametrize("key", sorted(required_keys()))
def test_required_key_is_not_left_empty(key: str) -> None:
    """An empty value raises the same way an absent key does."""
    for line in EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}="):
            assert stripped.split("=", 1)[1].strip(), f"{key} is required but ships empty"
            return
