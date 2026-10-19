"""Persist the optional placement objects layer.

Revision ID: 0008
Revises: 0007
"""

import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "pipeline_tasks",
        sa.Column("existing_objects_data_path", sa.String(1024), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("pipeline_tasks", "existing_objects_data_path")
