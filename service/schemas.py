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
    building_type_col: str | None = None
    building_service_col: str | None = None
    building_floors_col: str | None = None
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
    output_version: str | None = None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class ZoneSuggestionOut(BaseModel):
    """One PZZ zone missing from the built-in template, with the proposed match."""

    user_code: str
    user_name: str
    suggested_code: str | None
    suggested_name: str


class BuildingPzzCheckOut(BaseModel):
    """Result of submitting a building PZZ check: a task, or the step blocking it.

    Unlike the other submissions this one may legitimately not create a task — the
    zone review can need a descriptions file or a confirmation first — so ``task``
    is filled only when ``action`` is ``created``.

    ``chat_message`` is that outcome written out for a person, the counterpart of the
    field the object-zone-fit report carries: a chat client shows it rather than
    assembling its own text from ``action`` and ``suggestions``.
    """

    action: str
    narrative: str
    detail: str = ""
    next_step: str
    chat_message: str = ""
    suggestions: list[ZoneSuggestionOut] | None = None
    task: TaskOut | None = None


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
