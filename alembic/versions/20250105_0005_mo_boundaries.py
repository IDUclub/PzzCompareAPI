"""pipeline_tasks optional municipal boundary layer (overlap checks)

Revision ID: 0005
Revises: 0004
Create Date: 2025-01-05 00:00:00.000000+00:00

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "pipeline_tasks",
        sa.Column("mo_boundaries_data_path", sa.String(length=512), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("pipeline_tasks", "mo_boundaries_data_path")
