from datetime import datetime

from sqlalchemy import DateTime, Enum as SqlEnum, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .domain.task_state import TaskStatus  # noqa: F401 — re-exported for infrastructure consumers
from .time_utils import utc_now

__all__ = ["TaskStatus"]


class Base(DeclarativeBase):
    pass


class PipelineTask(Base):
    __tablename__ = "pipeline_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    cadastral_data_path: Mapped[str] = mapped_column(String(512))
    pzz_zones_data_path: Mapped[str] = mapped_column(String(512))
    pzz_zone_vri_labels_path: Mapped[str] = mapped_column(String(512))
    vri_classifier_path: Mapped[str] = mapped_column(String(512))
    include_pzz_check: Mapped[bool] = mapped_column(default=True)
    cadastral_vri_col: Mapped[str] = mapped_column(String(128))
    pzz_zone_code_col: Mapped[str] = mapped_column(String(128), default="Индекс_зоны")
    pzz_zone_name_col: Mapped[str] = mapped_column(String(128), default="Код_объекта")
    priority: Mapped[int] = mapped_column(Integer, default=1, index=True)
    status: Mapped[TaskStatus] = mapped_column(SqlEnum(TaskStatus), default=TaskStatus.queued)
    celery_task_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    result_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_pipeline_tasks_status_created_at", "status", "created_at"),
    )


class TaskEvent(Base):
    __tablename__ = "pipeline_task_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("pipeline_tasks.id", ondelete="CASCADE"), index=True)
    stage: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(64), index=True)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ConfigEntry(Base):
    __tablename__ = "config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    value: Mapped[str] = mapped_column(String(1024))
    py_type: Mapped[str] = mapped_column(String(64), default="str")


class TaskIdempotencyKey(Base):
    __tablename__ = "task_idempotency_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    task_external_id: Mapped[str] = mapped_column(
        ForeignKey("pipeline_tasks.external_id", ondelete="CASCADE"),
        unique=True,
        index=True,
    )
