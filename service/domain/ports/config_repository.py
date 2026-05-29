from __future__ import annotations

from typing import Protocol


class ConfigRepository(Protocol):
    def get_int(self, name: str, default: int, *, for_update: bool = False) -> int:
        ...

    def set_int(self, name: str, value: int) -> None:
        ...

    def get(self, name: str) -> tuple[str, str] | None:
        ...

    def set(self, name: str, value: str, py_type: str = "str") -> tuple[str, str]:
        ...
