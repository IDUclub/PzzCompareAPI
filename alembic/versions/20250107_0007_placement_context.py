"""pipeline_tasks placement NormGraph context path

Revision ID: 0007
Revises: 0006
Create Date: 2025-01-07 00:00:00.000000+00:00

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "pipeline_tasks",
        sa.Column("placement_context_path", sa.String(length=512), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("pipeline_tasks", "placement_context_path")
