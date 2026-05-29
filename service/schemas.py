from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import TaskStatus


class TaskCreate(BaseModel):
    cadastral_feature_collection: dict | None = None
    pzz_zones_feature_collection: dict | None = None
    pzz_zone_vri_labels: list[dict[str, Any]] | None = None
    vri_classifier: dict[str, Any] | list[dict[str, Any]] | None = None
    include_pzz_check: bool = True
    cadastral_vri_col: str
    pzz_zone_code_col: str = "Индекс_зоны"
    pzz_zone_name_col: str = "Код_объекта"
    priority: int = Field(default=1, ge=1, le=10)


class TaskOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    external_id: str
    cadastral_data_path: str
    pzz_zones_data_path: str
    priority: int
    status: TaskStatus
    include_pzz_check: bool
    cadastral_vri_col: str
    pzz_zone_code_col: str
    pzz_zone_name_col: str
    result_path: str | None
    error_text: str | None
    celery_task_id: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class TaskEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    stage: str
    status: str
    details: str | None
    created_at: datetime


class ConfigOut(BaseModel):
    name: str
    value: str
    py_type: str


class TaskListOut(BaseModel):
    items: list[TaskOut]
    total: int
    limit: int
    offset: int
