"""pipeline_tasks placement (service siting) columns

Revision ID: 0006
Revises: 0005
Create Date: 2025-01-06 00:00:00.000000+00:00

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_COLUMNS = (
    ("placement_vri_code", 32),
    ("placement_service_name", 256),
)


def upgrade() -> None:
    for name, length in _COLUMNS:
        op.add_column(
            "pipeline_tasks",
            sa.Column(name, sa.String(length=length), nullable=True),
        )


def downgrade() -> None:
    for name, _ in reversed(_COLUMNS):
        op.drop_column("pipeline_tasks", name)
