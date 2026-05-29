"""initial schema

Revision ID: 0001
Revises:
Create Date: 2025-01-01 00:00:00.000000+00:00

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_task_status_enum = sa.Enum(
    "queued",
    "waiting_capacity",
    "running",
    "finished",
    "failed",
    name="taskstatus",
)


def upgrade() -> None:
    op.create_table(
        "pipeline_tasks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("external_id", sa.String(length=64), nullable=False),
        sa.Column("cadastral_data_path", sa.String(length=512), nullable=False),
        sa.Column("pzz_zones_data_path", sa.String(length=512), nullable=False),
        sa.Column("pzz_zone_vri_labels_path", sa.String(length=512), nullable=False),
        sa.Column("vri_classifier_path", sa.String(length=512), nullable=False),
        sa.Column("include_pzz_check", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("cadastral_vri_col", sa.String(length=128), nullable=False),
        sa.Column("pzz_zone_code_col", sa.String(length=128), nullable=False, server_default="Индекс_зоны"),
        sa.Column("pzz_zone_name_col", sa.String(length=128), nullable=False, server_default="Код_объекта"),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", _task_status_enum, nullable=False, server_default="queued"),
        sa.Column("celery_task_id", sa.String(length=128), nullable=True),
        sa.Column("result_path", sa.String(length=1024), nullable=True),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_id", name="uq_pipeline_tasks_external_id"),
    )
    op.create_index("ix_pipeline_tasks_external_id", "pipeline_tasks", ["external_id"], unique=False)
    op.create_index("ix_pipeline_tasks_priority", "pipeline_tasks", ["priority"], unique=False)
    op.create_index(
        "ix_pipeline_tasks_status_created_at",
        "pipeline_tasks",
        ["status", "created_at"],
        unique=False,
    )

    op.create_table(
        "pipeline_task_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=False),
        sa.Column("stage", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("details", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["task_id"], ["pipeline_tasks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_pipeline_task_events_task_id", "pipeline_task_events", ["task_id"], unique=False)
    op.create_index("ix_pipeline_task_events_stage", "pipeline_task_events", ["stage"], unique=False)
    op.create_index("ix_pipeline_task_events_status", "pipeline_task_events", ["status"], unique=False)

    op.create_table(
        "config",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("value", sa.String(length=1024), nullable=False),
        sa.Column("py_type", sa.String(length=64), nullable=False, server_default="str"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_config_name"),
    )
    op.create_index("ix_config_name", "config", ["name"], unique=False)

    op.create_table(
        "task_idempotency_keys",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("task_external_id", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(
            ["task_external_id"],
            ["pipeline_tasks.external_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key", name="uq_task_idempotency_keys_key"),
        sa.UniqueConstraint("task_external_id", name="uq_task_idempotency_keys_external_id"),
    )
    op.create_index("ix_task_idempotency_keys_key", "task_idempotency_keys", ["key"], unique=False)
    op.create_index(
        "ix_task_idempotency_keys_task_external_id",
        "task_idempotency_keys",
        ["task_external_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_table("task_idempotency_keys")
    op.drop_table("config")
    op.drop_table("pipeline_task_events")
    op.drop_table("pipeline_tasks")
    _task_status_enum.drop(op.get_bind(), checkfirst=True)
