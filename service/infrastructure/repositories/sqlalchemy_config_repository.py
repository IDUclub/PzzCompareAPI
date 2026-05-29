from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from service.models import ConfigEntry


class SqlAlchemyConfigRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_int(self, name: str, default: int, *, for_update: bool = False) -> int:
        q = select(ConfigEntry).where(ConfigEntry.name == name)
        if for_update:
            q = q.with_for_update()
        row = self.session.execute(q).scalar_one_or_none()

        if row is None:
            row = ConfigEntry(name=name, value=str(default), py_type="int")
            self.session.add(row)
            self.session.flush()

        return int(row.value)

    def set_int(self, name: str, value: int) -> None:
        row = self.session.execute(
            select(ConfigEntry).where(ConfigEntry.name == name)
        ).scalar_one_or_none()

        if row is None:
            row = ConfigEntry(name=name, value=str(value), py_type="int")
            self.session.add(row)
            return

        row.value = str(value)
        row.py_type = "int"

    def get(self, name: str) -> tuple[str, str] | None:
        row = self.session.execute(
            select(ConfigEntry).where(ConfigEntry.name == name)
        ).scalar_one_or_none()
        if row is None:
            return None
        return row.value, row.py_type

    def set(self, name: str, value: str, py_type: str = "str") -> tuple[str, str]:
        row = self.session.execute(
            select(ConfigEntry).where(ConfigEntry.name == name)
        ).scalar_one_or_none()

        if row is None:
            row = ConfigEntry(name=name, value=value, py_type=py_type)
            self.session.add(row)
            return value, py_type

        row.value = value
        row.py_type = py_type
        return row.value, row.py_type
