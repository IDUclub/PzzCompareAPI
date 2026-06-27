# API сервиса классификации ВРИ — гайд для фронтенда

## Что это и зачем

Бэкенд классифицирует кадастровые участки против Правил Землепользования и Застройки (ПЗЗ): для каждого участка определяет «в правильной ли зоне он стоит» и подбирает наиболее подходящий вид разрешённого использования (ВРИ). Поверх классификации есть **чат-режим**: пользователь задаёт вопрос — сервис стримит текстовый ответ ИИ и сохраняет диалог.

Классификация **асинхронная** — все ручки создания задач возвращают сразу с `status: "queued"`, реальный расчёт идёт в фоне (от нескольких секунд до часов, зависит от объёма данных и нагрузки LLM).

## Базовое

| | |
|---|---|
| **Base URL** | `http://<host>:8000` |
| **Swagger UI** | `GET /docs` |
| **Auth** | `Authorization: Bearer <jwt>` обязателен для `/scenarios/*` (пробрасывается в urban_api) и для всех чат-ручек `*/chat/stream` (под этим пользователем пишется история чата). Остальные файловые ручки токен не требуют. |
| **Content-Type** | `multipart/form-data` для создания задач (файлы), `application/json` для остального |

---

## Сценарии использования

### Сценарий A: «У юзера свои файлы»
Юзер вручную грузит геоданные кадастровых участков (+ опционально PZZ-зон) → классификация. Принимается GeoJSON, а также GeoPackage/GML/KML/GeoParquet (см. [B1](#b1-post-taskspzz-check--полная-проверка-с-pzz)).

### Сценарий B: «У юзера есть scenario_id в urban_api»
Бэкенд сам тянет данные из urban_api по `scenario_id` + `(year, source)` → классификация. Никаких ручных загрузок.

### Сценарий C: «Чат с ИИ»
Поверх A или B — пользователь задаёт вопрос (`user_query`), сервис классифицирует, **стримит текстовый ответ** и сохраняет диалог в ChatStorage. Ручки `*/chat/stream`, см. [раздел H](#h-чат-ответ-llm-поверх-классификации-sse).

---

## Состояния задачи

```
queued ──► running ──► finished     (успех)
   │           │
   │           └────►  failed       (ошибка пайплайна)
   │
   └► waiting_capacity ──► queued   (если capacity сейчас исчерпан)
```

`finished` и `failed` — терминальные. Активные: `queued` / `waiting_capacity` / `running`.

---

# Эндпоинты

## A. System

### `GET /health`
Простой liveness. Не требует ничего.
```json
{"status": "ok"}
```

### `GET /readiness`
Проверяет БД + Redis. Возвращает `503` если что-то недоступно.

### `GET /docs`
Swagger UI с полной OpenAPI-схемой.

---

## B. Создание задачи (выбрать ОДИН способ)

### B1. `POST /tasks/pzz-check` — полная проверка с PZZ

**Когда использовать:** у юзера есть и кадастр, и PZZ-зоны в виде GeoJSON.

**Тело запроса (multipart/form-data):**

| Поле | Тип | Обязательно | Описание |
|------|-----|-------------|----------|
| `cadastral_feature_collection_file` | File | да | Кадастровые участки. GeoJSON в **EPSG:4326**, либо любой гео-формат (см. ниже) |
| `pzz_zones_feature_collection_file` | File | да | PZZ-зоны. GeoJSON в **EPSG:4326**, либо любой гео-формат |
| `pzz_zone_vri_labels_file` | File | – | Свой JSON с описанием зон (если нет — используется дефолт) |
| `vri_classifier_file` | File | – | Свой классификатор Росреестра (если нет — дефолт) |
| `cadastral_vri_col` | string | да | Имя поля в кадастре с текстом ВРИ (например `"Вид разреш"`) |
| `pzz_zone_code_col` | string | да | Имя поля в PZZ с кодом зоны (например `"Индекс_зоны"`) |
| `pzz_zone_name_col` | string | да | Имя поля в PZZ с названием зоны |
| `priority` | int 1–10 | – | По умолчанию 1 |
| `force_recompute` | bool | – | По умолчанию false. См. секцию «Idempotency» |
| `Idempotency-Key` | string (header или form) | – | Опциональный ключ дедупликации |

**Ответ (200):**
```json
{
  "id": 42,
  "external_id": "a1b2c3d4e5f6...",
  "status": "queued",
  "priority": 1,
  "include_pzz_check": true,
  "cadastral_vri_col": "Вид разреш",
  "created_at": "2026-05-22T10:15:30Z",
  "started_at": null,
  "finished_at": null,
  "celery_task_id": "...",
  "result_path": null,
  "error_text": null
}
```

**Что делать дальше:** запомнить `external_id`, поллить `GET /tasks/{external_id}`.

**Поддерживаемые форматы файлов** (для `cadastral_*` и `pzz_zones_*`): `.geojson` / `.json`
(должны быть в EPSG:4326), а также `.gpkg` (GeoPackage), `.gml`, `.kml`, `.geoparquet` / `.parquet`.
Не-GeoJSON форматы бэкенд читает через geopandas, **репроецирует в EPSG:4326** и хранит
как GeoJSON. CRS берётся из самого файла; если в файле CRS не задан — данные считаются уже
в EPSG:4326.

**Возможные ошибки:**
- `422` — GeoJSON некорректный или не в EPSG:4326
- `415` — неподдерживаемое расширение файла
- `400` — гео-файл не читается / не конвертируется
- `413` — файл больше лимита (200 МБ)

---

### B2. `POST /tasks/classify-only` — только классификация (без PZZ)

**Когда:** есть только кадастр, нет PZZ-зон. Будет классификация против федерального классификатора Росреестра без spatial-overlay.

**Тело:** как у `/pzz-check`, но **без** `pzz_zones_feature_collection_file`, `pzz_zone_code_col`, `pzz_zone_name_col`.

**Ответ:** такой же `TaskOut`.

---

### B3. `POST /scenarios/{scenario_id}/classify` — классификация по сценарию urban_api

**Когда:** юзер выбрал свой проект в urban_api, хочет классифицировать жилые дома.

**URL:** `POST /scenarios/{scenario_id}/classify`

**Headers:**
```
Authorization: Bearer <jwt>     ← обязателен, пробрасывается в urban_api
```

**Тело (multipart/form-data):**

| Поле | Тип | Обязательно | По умолчанию |
|------|-----|-------------|--------------|
| `year` | int | да | — |
| `source` | string | да | `User` / `OSM` / `PZZ` |
| `physical_object_type_id` | int | – | 4 (жилые дома) |
| `priority` | int 1–10 | – | 1 |
| `force_recompute` | bool | – | false |
| `Idempotency-Key` | string | – | опционально |

**Что делает бэк:**
1. Проверяет в urban_api что для `(year, source)` есть данные → `/api/v1/scenarios/{id}/functional_zone_sources`
2. Скачивает зоны → `/functional_zones?year=&source=`
3. Скачивает объекты → `/physical_objects_with_geometry?physical_object_type_id=...`
4. Создаёт задачу пайплайна с этими данными как входами

**Ответ (200):** такой же `TaskOut` как у `/tasks/pzz-check`.

**Ошибки:**
- `422` с подробностями если `(year, source)` отсутствует:
  ```json
  {
    "detail": {
      "error": "no functional zones for the requested (year, source)",
      "scenario_id": 843,
      "requested": {"year": 2026, "source": "PZZ"},
      "available": [{"year": 2026, "source": "User"}]
    }
  }
  ```
- `502` — urban_api недоступен
- `503` — `URBAN_API_BASE_URL` не настроен на бэкенде

---

### B4. Защищённые ручки результата для сценариев

Для задач, созданных через `/scenarios/{scenario_id}/classify`, фронт должен использовать scenario-scoped URL и передавать тот же `Authorization: Bearer <jwt>`.

| Что нужно | URL |
|----------|-----|
| статус задачи | `GET /scenarios/{scenario_id}/tasks/{external_id}` |
| GeoJSON-результат | `GET /scenarios/{scenario_id}/tasks/{external_id}/result` |
| UI/чат-агрегация | `GET /scenarios/{scenario_id}/tasks/{external_id}/object-zone-fit?group_by=zone\|object` |

Бэкенд сначала проверяет доступ к `scenario_id` в urban_api по токену, затем проверяет, что `external_id` действительно относится к этому сценарию. Общие `/tasks/{external_id}/*` остаются для задач, созданных загрузкой файлов.

---

## C. Опрос задачи

### `GET /tasks/{external_id}`

**Что:** возвращает текущее состояние задачи. Опрашивайте раз в **3–5 секунд** пока статус не станет терминальным (`finished` или `failed`).

**Ответ:**
```json
{
  "id": 42,
  "external_id": "a1b2...",
  "status": "running",
  "priority": 1,
  "created_at": "...",
  "started_at": "2026-05-22T10:15:35Z",
  "finished_at": null,
  "result_path": null,
  "error_text": null,
  "celery_task_id": "..."
}
```

Возможные значения `status`: `queued` | `waiting_capacity` | `running` | `finished` | `failed`.

**Если `status: "failed"`** — взять `error_text` для показа юзеру.
**Если `status: "finished"`** — переходить к разделу D.

### `GET /tasks/{external_id}/events`

**Что:** журнал событий задачи. Полезно для отладки.

**Ответ:**
```json
[
  {"id": 1, "stage": "queue", "status": "enqueued", "details": "celery_id=...", "created_at": "..."},
  {"id": 2, "stage": "pipeline", "status": "start", "details": null, "created_at": "..."},
  {"id": 3, "stage": "pipeline", "status": "finished", "details": "...", "created_at": "..."}
]
```

### `GET /tasks_list?status=&limit=&offset=`

**Что:** список всех задач с фильтром по статусу и пагинацией.

```json
{
  "items": ["TaskOut", "TaskOut", "..."],
  "total": 87,
  "limit": 20,
  "offset": 0
}
```

---

## D. Получение результата (когда status=finished)

### D1. `GET /tasks/{external_id}/result`

**Что:** сырой GeoJSON-результат. Скачивается как файл (`Content-Type: application/geo+json`).

Для каждого Feature в `properties` будут поля:
- `ВРИ_ЕГРН` — исходный текст ВРИ
- `Код фактической зоны нахождения кадастра`
- `Название фактической зоны нахождения кадастра`
- `Вердикт_ПЗЗ` — `allowed_main` / `allowed_conditional` / `allowed_auxiliary` / `not_allowed` / `unclear` / `no_actual_zone` / `no_zone_metadata` / `classifier_only`
- `Причина` — текстовое объяснение
- `Подобранный_ВРИ` + `Код_подобранного_ВРИ`
- `Топ5_возможных_ВРИ` — для not_allowed: 5 ближайших разрешённых вариантов

Эта ручка для **карты / GIS-клиента** — отдаёт геометрию + атрибуты.

### D2. `GET /tasks/{external_id}/object-zone-fit?group_by=zone|object`

**Самая полезная ручка для UI / чат-бота.** Возвращает структурированный отчёт без геометрии.

**`group_by=zone` (по умолчанию):**
```json
{
  "task_external_id": "abc...",
  "group_by": "zone",
  "summary": {
    "total": 60,
    "in_correct_zone": 55,
    "in_wrong_zone": 3,
    "unclear": 2
  },
  "chat_message": "Проверено объектов: 60 в 5 зонах.\nВ подходящих зонах: 55. Не в своих зонах: 3. Без чёткой оценки: 2.\n\nЗона «Транспортная зона»: 2 из 2 не в своей зоне.\n    Жилой дом не разрешён в зоне транспорта\nЗона «Жилая зона»: все 53 в порядке.",
  "zones": [
    {
      "zone_type_id": "6",
      "zone_name": "Транспортная зона",
      "pzz_summary": {
        "mapping_status": "no_mapping",
        "allowed_construction_summary": null
      },
      "summary": {"total": 2, "in_correct_zone": 0, "in_wrong_zone": 2, "unclear": 0},
      "objects": [
        {
          "feature_index": 0,
          "vri_text": "Жилой дом, 5 этажей",
          "zone_type_id": "6",
          "zone_name": "Транспортная зона",
          "verdict": "not_allowed",
          "is_in_correct_zone": false,
          "fit": "wrong",
          "reason": "Жилой дом не разрешён в зоне транспорта...",
          "matched_vri_name": null,
          "matched_vri_code": null
        }
      ]
    }
  ]
}
```

Возможные значения `fit`: `"correct"` | `"wrong"` | `"unclear"`.

**`group_by=object`:** то же, но плоский список `objects: [...]` без группировки.

**Ключевые поля для UI:**

| Поле | Что показать |
|------|--------------|
| `summary.total` / `in_correct_zone` / `in_wrong_zone` | KPI-плашки |
| `chat_message` | Готовый текст для чат-бота (plain-text) |
| `zones[].zone_name` | Название зоны (русское) |
| `zones[].pzz_summary.allowed_construction_summary` | Справка «что можно строить в этой зоне» |
| `zones[].objects[].feature_index` | Привязка к Feature в GeoJSON (D1) для подсветки на карте |
| `zones[].objects[].fit` | Цвет маркера: correct=зелёный, wrong=красный, unclear=жёлтый |
| `zones[].objects[].reason` | Тултип или модалка при клике на объект |

---

## E. Справка по зонам (без классификации)

### `GET /scenarios/{scenario_id}/zones-info?year=&source=`

**Когда:** юзер хочет «просто посмотреть какие зоны есть в проекте и что в них можно строить» — без запуска классификации.

**Headers:** `Authorization: Bearer <jwt>`

**Ответ:**
```json
{
  "scenario_id": 843,
  "year": 2026,
  "source": "User",
  "total": 12,
  "items": [
    {
      "functional_zone_id": 7711068,
      "zone_type_id": 7,
      "zone_type_name": "Общественно-деловая зона",
      "name": null,
      "year": 2026,
      "source": "User",
      "properties": {},
      "pzz_summary": {
        "mapping_status": "ok",
        "mapping_confidence": "high",
        "mapping_note": "...",
        "db_zone_nickname": "Общественно-деловая зона",
        "source_pzz_zone_codes": ["ОД-1", "ОД-2"],
        "allowed_construction_summary": "Основные: 4.1 — Деловое управление; 4.6 — Общественное питание; ...",
        "main_vri": [],
        "conditional_vri": [],
        "auxiliary_vri": []
      }
    }
  ]
}
```

Возможные значения `mapping_status`:
- `"ok"` — есть справка ПЗЗ, показывайте всё
- `"low_confidence"` — справка есть, но качество ниже (можно показать с пометкой «приблизительно»)
- `"no_mapping"` — нет соответствия в нашей базе, поля `*_vri` и `*_summary` будут `null`

---

## F. Управление задачей

### `POST /tasks/{external_id}/recompute`

**Когда:** задача в `finished` или `failed`, юзер жмёт «пересчитать». Создаёт новую задачу с тем же `external_id` и теми же входами (файлы не перезагружаются).

**Ответ:** обновлённый `TaskOut` со `status: "queued"` и новым `celery_task_id`.

**Ошибки:**
- `409` — задача в активном статусе (нельзя пересчитать пока работает)
- `404` — задачи нет

### `DELETE /tasks/{external_id}`

**Когда:** юзер хочет отменить активную задачу.

- Если `queued`/`waiting_capacity` — отменяется тихо
- Если `running` — посылается SIGTERM воркеру
- Если `finished`/`failed` — `409 Conflict`

**Ответ:** обновлённый `TaskOut` со `status: "failed"`, `error_text: "Cancelled by client"`.

---

## G. Идемпотентность и форс-пересчёт

Чтобы избежать дублей при повторных кликах юзера / refresh страницы — фронт может передавать **`Idempotency-Key`** (в header или form-поле). Бэкенд хранит маппинг `key → task` и при повторной отправке того же ключа возвращает существующую задачу.

```js
// Например — хэш файлов как ключ
const key = await sha256(file1.bytes + file2.bytes);
fetch('/tasks/pzz-check', {
  method: 'POST',
  headers: { 'Idempotency-Key': key },
  body: formData,
});
```

**Поведение:**

| Сценарий | Что вернётся |
|----------|--------------|
| Первый запрос с ключом X | Создаётся новая задача |
| Повтор с X (задача `running`) | Та же задача |
| Повтор с X (задача `finished`) | Та же задача (без пересчёта) |
| Повтор с X (задача `finished`) + `force_recompute=true` | Перезапускается тот же `external_id` |
| Повтор с X (задача `failed`) + `retry_failed=true` ИЛИ `force_recompute=true` | Перезапускается |

Ключи **изолированы по типу endpoint'а** — один и тот же `Idempotency-Key` на `/tasks/pzz-check` и `/tasks/classify-only` создаст две разные задачи (что обычно и нужно).

---

## H. Чат-ответ (LLM поверх классификации, SSE)

Две ручки одним вызовом запускают классификацию, **дожидаются её завершения**, затем
**стримят разговорный ответ** LLM на запрос пользователя (`user_query`), опираясь на отчёт
object-zone-fit. Параллельно история диалога сохраняется в сервис **ChatStorage** под
пользователем из токена.

| Флоу | Ручка |
|------|-------|
| По сценарию urban_api | `POST /scenarios/{scenario_id}/chat/stream` |
| По загруженным файлам | `POST /tasks/chat/stream` |

**Auth:** `Authorization: Bearer <jwt>` **обязателен** (без токена история не пишется —
`user_id` берётся из токена на стороне ChatStorage; `project_id` бэкенд сам тянет из urban_api).

**Тело (multipart/form-data):** как у соответствующего `classify` / `pzz-check`, плюс:

| Поле | Тип | Обязательно | Описание |
|------|-----|-------------|----------|
| `user_query` | string | да | Текст запроса пользователя (как в gMART) |
| `chat_id` | string | – | UUID существующего чата. Если не передан — бэкенд создаст новый и пришлёт `service_event`/`chat_created` |
| `group_by` | `zone`\|`object` | – | По умолчанию `zone` |
| `model` | string | – | Имя модели Ollama. По умолчанию — `CHAT_MODEL`/`GENERATE_MODEL` бэкенда |
| `temperature` | float | – | Переопределить температуру модели |

**Транспорт:** Server-Sent Events. Используйте **fetch-based SSE-клиент** — нативный
`EventSource` не умеет POST/multipart и не ставит `Authorization`.

**Поток событий (в порядке).** События чат-части идут в формате gMART — JSON-конверт
`{ "type", "content" }` в поле `data`:

| `data.type` (или SSE event) | data | Когда |
|-------|------|-------|
| `task` | `TaskOut` | сразу, дескриптор созданной задачи (сохраните `external_id`) |
| `task_event` | событие пайплайна | по мере выполнения |
| `status` | `TaskOut` | при смене статуса |
| `object_zone_fit` | отчёт (см. D2) | когда задача `finished` |
| `file` | `{ "name", "url", "download_url", "filename", "mime_type", "source_service" }` | ссылка на геослой-результат (см. ниже) |
| `service_event` | `{ "event_type": "storage_event", "event": { "storage_event_type": "chat_created", "chat_id", "chat_title" } }` | только если `chat_id` не был передан — **сохраните `chat_id`** |
| `chunk` | `{ "text": "...", "done": false }` | дельты ответа LLM; финальный `{ "text": "", "done": true }` — конец ответа |
| `error` | `{ "message", "stage" }` | не фатально (сбой LLM/ChatStorage) — поток продолжается |
| `done` | `{ "status", "chat_id" }` | терминал, поток закрывается |

Собирайте ответ ассистента, конкатенируя `content.text` из событий `chunk` (до `done: true`).
Полный ответ бэкенд сам сохранит в ChatStorage как `role: "assistant"`, а `user_query` —
как `role: "user"`.

### Геослои как ссылки (`file`)

Большие GeoJSON **не** приходят инлайном — вместо этого приходит событие `file` со ссылками.
Поле `role` различает результат и входные слои:

```json
{ "type": "file", "content": {
  "name": "classified_result",          // или input_cadastral / input_zones
  "role": "result",                      // "result" | "input"
  "url": "https://<api>/files/result/<external_id>",   // долговечная ссылка (не протухает)
  "download_url": "https://<minio>/...?X-Amz-Signature=...", // presigned для мгновенной выгрузки (может быть null)
  "filename": "<external_id>.geojson",
  "mime_type": "application/geo+json",
  "source_service": "PZZ Pipeline Service"
} }
```

Какие `file`-события приходят:
- `role: "result"` — итоговый классифицированный слой (когда задача `finished`). Приходит во всех
  стримах (чат и обычные `*/classify/stream`, `*/pzz-check/stream`).
- `role: "input"` — **загруженные** входные слои (`input_cadastral`, `input_zones`). Приходят
  **только в upload-флоу** (`/tasks/chat/stream`, `/tasks/pzz-check/stream`,
  `/tasks/classify-only/stream`), сразу в начале (можно качать, не дожидаясь завершения).
  В сценарном флоу их нет (входные данные тянутся из urban_api).

Как пользоваться ссылками:
- **мгновенная** выгрузка → `download_url` (если не `null`);
- **постоянная** ссылка (карточка в истории, шаринг) → `url` — это `GET /files/{slot}/{external_id}`
  (`slot` ∈ `result` / `cadastral` / `zones`), которая на каждый заход редиректит (307) на свежий
  presigned MinIO. Не протухает, авторизация не нужна, большой файл качается прямо из MinIO.

В ChatStorage сохраняется только **result**-ссылка — как `kind: "file"` часть сообщения ассистента
(`payload.url` = долговечный `url`). Входные слои в историю не пишутся (приходят только в стриме).
`download_url` нигде не сохраняется (он временный).

**Пример (frontend):**
```js
const res = await fetch(`/scenarios/843/chat/stream`, {
  method: "POST",
  headers: { Authorization: `Bearer ${token}` },
  body: formData, // user_query, year, source, [chat_id], ...
});
const reader = res.body.getReader();
// ...парсинг SSE: на 'chat_created' сохранить chat_id, на 'token' дописать в пузырь ответа
```

История чатов (список, открытие, удаление) живёт в самом ChatStorage — см. его собственный
фронт-гайд; этот сервис только пишет в него user/assistant сообщения.

---

# Типовые сценарии работы фронта

## Сценарий 1: Юзер грузит файлы и ждёт результат

```
1. Юзер открывает форму, заполняет файлы и поля
2. Фронт: POST /tasks/pzz-check (или /tasks/classify-only)
   ← TaskOut { external_id: "abc...", status: "queued" }
3. Фронт показывает прелоадер «Задача в очереди, идёт расчёт...»
4. Фронт: setInterval(() => fetch(`/tasks/abc...`), 3000)
   ← status: "queued"     → показывает «В очереди»
   ← status: "running"    → показывает «Идёт расчёт»
   ← status: "finished"   → стопает поллинг, переходит к шагу 5
   ← status: "failed"     → показывает error_text юзеру
5. Параллельно:
   - Карта: GET /tasks/abc/result    → отрисовка GeoJSON
   - Отчёт: GET /tasks/abc/object-zone-fit?group_by=zone  → KPI + список зон
   - Чат:   chat_message из ответа выше
```

## Сценарий 2: Юзер открывает проект из urban_api

```
1. Юзер выбрал scenario_id=843, year=2026, source=User
2. Фронт (опционально): GET /scenarios/843/zones-info?year=2026&source=User
   ← список зон со справками pzz_summary (для side-panel «зоны проекта»)
3. Юзер жмёт «Классифицировать»
   Фронт: POST /scenarios/843/classify { year, source }
   ← TaskOut со status: "queued"
4. Фронт поллит GET /scenarios/843/tasks/{external_id} с Authorization
5. После finished:
   - Карта: GET /scenarios/843/tasks/{external_id}/result
   - Отчёт: GET /scenarios/843/tasks/{external_id}/object-zone-fit?group_by=zone
```

## Сценарий 3: Юзер хочет пересчитать готовую задачу

```
1. Фронт: POST /tasks/{external_id}/recompute
   ← TaskOut со status: "queued"
2. Возвращается к поллингу
```

## Сценарий 4: Чат с ИИ по проекту

```
1. Юзер вводит вопрос (user_query) на странице чата
2. Фронт (fetch-based SSE): POST /scenarios/{id}/chat/stream  (или /tasks/chat/stream)
   { Authorization: Bearer <jwt> }
3. Ловит события:
   ← task               → сохранить external_id
   ← object_zone_fit    → KPI/отчёт
   ← file (result)      → ссылка на слой для карты/скачивания
   ← service_event      → сохранить chat_id (если новый чат)
   ← chunk*             → дописывать текст ответа ассистента (до done:true)
   ← done               → закрыть поток
4. Следующий вопрос в том же диалоге — передать chat_id (история подтянется).
```
---
