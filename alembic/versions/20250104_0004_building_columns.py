"""pipeline_tasks building-column names (uploaded building PZZ check)

Revision ID: 0004
Revises: 0003
Create Date: 2025-01-04 00:00:00.000000+00:00

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_COLUMNS = ("building_type_col", "building_service_col", "building_floors_col")


def upgrade() -> None:
    for name in _COLUMNS:
        op.add_column(
            "pipeline_tasks",
            sa.Column(name, sa.String(length=128), nullable=True),
        )


def downgrade() -> None:
    for name in _COLUMNS:
        op.drop_column("pipeline_tasks", name)
