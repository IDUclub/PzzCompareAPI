# PZZ Pipeline Service

Сервис классификации кадастровых объектов по правилам землепользования и
застройки (**ПЗЗ**). Сравнивает функциональные зоны проекта с объектами,
которые на них стоят, и сообщает, какие объекты находятся **не в своей зоне**,
с человекочитаемым объяснением для чат-бота.

Построен вокруг готового пайплайна классификации (`pipeline_modules/`),
обёрнутого в продакшн-сервис: HTTP API, фоновый воркер, объектное хранилище и
MCP-сервер для AI-агентов.

---

## Что умеет

Три сценария использования (соответствуют пунктам ТЗ):

1. **Сравнение зон с объектами** — определить, какие объекты стоят не в своей
   функциональной зоне (пока в фокусе — жилые объекты).
2. **Справка по зонам** — по проекту отдать список зон с текстовым описанием
   «что можно / нельзя строить».
3. **Ответ чат-бота** — готовый plain-text с перечнем неуместных объектов.

Два способа подать данные:

- **Сценарный флоу** (для агентов): данные тянутся из IDU `urban_api` по
  `scenario_id` + `(year, source)`. Токен пользователя форвардится в urban_api.
- **Файловый флоу**: пользователь загружает GeoJSON напрямую
  (`/tasks/pzz-check`, `/tasks/classify-only`).

---

## Архитектура

```
                ┌─────────────┐      ┌──────────────┐
   агент ─MCP──▶│  mcp_server │─REST▶│   FastAPI    │
                └─────────────┘      │   (api)      │
   фронт ─REST───────────────────────▶              │
                                     └──────┬───────┘
                                            │ enqueue (Redis)
                                     ┌──────▼───────┐   ┌──────────┐
                                     │ Celery worker│──▶│ pipeline │
                                     │   + beat     │   │ _modules │
                                     └──────┬───────┘   └──────────┘
                                            │
                          Postgres ◀────────┴────────▶ MinIO (inputs/outputs)
```

Слои внутри `service/`:

| Каталог | Назначение |
|---|---|
| `api/` | HTTP-эндпоинты (classifier, scenarios, tasks, system) |
| `application/` | use-cases (create/start/finish task) — оркестрация |
| `domain/` | бизнес-правила, порты (контракты репозиториев), состояния задач |
| `infrastructure/` | реализации: БД-репозитории, MinIO, urban_api клиент, PZZ-маппинг, **раннер пайплайна** |
| `mcp_server/` | MCP-сервер для агентов (отдельный процесс, тонкий клиент к API) |

`pipeline_modules/` — **сторонний пайплайн** классификации (geopandas + NLP +
LLM). Сервис вызывает его через порт `PipelineRunner`
(`infrastructure/runners/pipeline_runner.py`) — in-process или в subprocess, с
прозрачной загрузкой/выгрузкой входов и результатов из MinIO.

---

## Технологии

- Python 3.11, FastAPI, Celery, SQLAlchemy + Alembic
- PostgreSQL, Redis, MinIO (S3-совместимое хранилище)
- FastMCP 3.x (MCP-сервер), Prometheus-метрики
- Docker / Docker Compose

---

## Быстрый старт (локально)

Требуется заполненный `.env.development` (см. `.env.example`) с доступами к БД,
Redis, MinIO, LLM-бэкенду и (для сценарного флоу) urban_api.

```bash
docker compose -f docker-compose.yml up -d --build
```

Поднимет 6 сервисов: `postgres`, `redis`, `api` (миграции на старте),
`worker`, `beat`, `mcp`.

| Сервис | Адрес |
|---|---|
| API | http://localhost:8000 (OpenAPI: `/docs`) |
| MCP | http://localhost:8765/mcp (health: `/health`) |
| Метрики воркера | http://localhost:9100/metrics |
| Postgres / Redis | 5432 / 6379 |

Свернуть: `docker compose -f docker-compose.yml down`.

---

## Эндпоинты (основное)

**Файловый флоу**
- `POST /tasks/pzz-check` — полная проверка ПЗЗ (кадастр + зоны)
- `POST /tasks/classify-only` — только классификация ВРИ по классификатору
- `POST /tasks/chat/stream` — проверка + стрим разговорного ответа LLM (SSE, требует Bearer)
- `GET /tasks/{id}` · `GET /tasks_list` · `GET /tasks/{id}/result`
- `GET /files/{slot}/{id}` — долговечная ссылка на геослой (`slot`: `result`/`cadastral`/`zones`; 307 → presigned MinIO)
- `GET /tasks/{id}/object-zone-fit?group_by=zone|object` — структурированный отчёт + `chat_message`
- `GET /tasks/{id}/events` · `DELETE /tasks/{id}` · `POST /tasks/{id}/recompute`

Загрузки кадастра/зон принимают GeoJSON, а также GeoPackage `.gpkg`, GML, KML и
GeoParquet — не-GeoJSON форматы конвертируются в GeoJSON (EPSG:4326) на входе.

**Сценарный флоу** (требует `Authorization: Bearer <jwt>`)
- `POST /scenarios/{id}/classify` — запуск по данным urban_api
- `POST /scenarios/{id}/chat/stream` — запуск + стрим разговорного ответа LLM (SSE)
- `GET /scenarios/{id}/zones-info` — зоны + справка «что можно строить»
- `GET /scenarios/{id}/tasks/{external_id}` (+ `/result`, `/object-zone-fit`, `/events`)
- `DELETE` / `POST .../recompute`

Чат-ручки `*/chat/stream` дожидаются завершения классификации, затем стримят (в формате gMART:
конверт `{type, content}`) `object_zone_fit` → `service_event/chat_created` (если не передан
`chat_id`) → `chunk`* → `done`, и сохраняют диалог (user + assistant) в **ChatStorage**.
Разговорный ответ генерирует Ollama `/api/chat` (`OLLAMA_BASE_URL`); модель — параметр запроса
`model` (дефолт `CHAT_MODEL`/`GENERATE_MODEL`).

Большой GeoJSON-результат в чат-стриме отдаётся **ссылкой** (событие `file`), а не инлайном:
долговечный `url = /files/result/{id}` (307 → свежий presigned MinIO, не протухает) сохраняется в
ChatStorage как `file`-часть сообщения; временный `download_url` — для мгновенной выгрузки.
Настройки: `PUBLIC_BASE_URL` (абсолютные ссылки), `GEO_LAYER_URL_TTL_SECONDS`.

**Системное**: `GET /health`, `GET /readiness`, `GET /metrics`

Подробное руководство для фронтенда — [`docs/frontend-api-guide.md`](docs/frontend-api-guide.md).

---

## MCP (для AI-агентов)

MCP-сервер (`service/mcp_server/`) — отдельный процесс на FastMCP, тонкий
клиент поверх REST API. Позволяет AI-агенту (Claude и др.) запускать
классификацию и читать отчёты как набор инструментов, без знания внутренностей
сервиса. Транспорт — streamable-HTTP на `:8765/mcp` (`MCP_API_BASE_URL`
указывает на api).

### Инструменты

**Сценарные** (`tools/scenarios.py`):

| Инструмент | Назначение |
|---|---|
| `classify_scenario` | Запустить классификацию по данным urban_api → `external_id` |
| `classify_scenario_and_wait` | То же + ожидание с прогрессом, сразу возвращает отчёт |
| `get_scenario_classification_status` | Статус задачи (`queued/running/finished/failed`) |
| `get_scenario_classification_report` | Отчёт «кто не в своей зоне» + готовый `chat_message` |
| `get_scenario_zones_info` | Зоны проекта + справка «что можно строить» |
| `recompute_scenario_classification` | Принудительный пересчёт |

**Файловые** (`tools/tasks.py`): `submit_pzz_check_task`,
`submit_classify_only_task`, `get_task_status`, `list_tasks`,
`get_task_events`, `get_task_result`, `cancel_task`, `recompute_task`.

### Параметры

Агент передаёт содержательные параметры **явно**: `scenario_id`, `year`,
`source`, `physical_object_type_id` (4 = жилые), `group_by`, `external_id`
и т.п. `scenario_id` указывает, с каким проектом/сценарием работать.

**Не** является аргументом инструмента только аутентификация: Bearer-токен
пользователя берётся из заголовка `Authorization` и форвардится в urban_api.

### Типовой диалог

```
get_scenario_zones_info(scenario_id, year, source)          # справка + проверка доступа/токена
classify_scenario(scenario_id, year, source, type_id) → external_id
poll get_scenario_classification_status(scenario_id, external_id) → finished
get_scenario_classification_report(scenario_id, external_id) # отдать chat_message пользователю
```

### Обработка токена

Если urban_api отвергает токен, инструмент возвращает
`-32002 AUTH_TOKEN_EXPIRED` — агент должен запросить свежий Bearer у
фронтенда/пользователя и **не** повторять со старым.

### Подключение (Claude Desktop)

Через мост `mcp-remote` (stdio → HTTP):

```json
{ "command": "cmd", "args": ["/c", "npx", "-y", "mcp-remote", "http://localhost:8765/mcp"] }
```

Смоук-тест end-to-end: `scripts/test_mcp_scenario.py` (нужен `URBAN_API_TOKEN`,
опц. `SCENARIO_ID` / `SCENARIO_YEAR` / `SCENARIO_SOURCE`).

---

## Аутентификация

Сценарные эндпоинты (`/scenarios/*`) требуют `Authorization: Bearer <jwt>`.
Токен форвардится в urban_api и, опционально, **проверяется локально** по
образцу IDUclub (Keycloak JWT через JWKS, см. `service/auth/`):

- `AUTH_VERIFY=false` (по умолчанию) — токен принимается без проверки подписи
  (dev, либо когда его уже проверил вышестоящий шлюз);
- `AUTH_VERIFY=true` + `AUTH_SERVER_URL=https://<host>/realms/<realm>` —
  проверяются подпись (RS256 по JWKS), issuer и (опц.) audience; отвергнутый
  токен → `401`.

Параметры: `AUTH_SERVER_URL`, `AUTH_CLIENT_ID`, `AUTH_VERIFY_AUD`,
`AUTH_VALID_AUDIENCES`, кеши `AUTH_JWKS_CACHE_TTL` / `AUTH_USER_CACHE_TTL`.

## Метрики

Метрики задач (`queue_wait_seconds`, `task_run_seconds`, `task_fail_total`,
`task_retry_total`) пишутся в процессе воркера, поэтому экспонируются **им** на
`:9100` — отдельным таргетом от API `:8000/metrics`. Prometheus должен скрейпить
оба.

---

## Тесты

```bash
pip install -r requirements.txt
pytest
```

Тесты герметичны (sqlite, dummy-окружение в `tests/conftest.py`) — живой
Postgres/Redis не нужен. Пайплайн-тесты требуют установленных зависимостей
пайплайна (geopandas, nltk и пр.).

---

## Деплой

CI-пайплайн [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)
(`workflow_dispatch`) собирает образ, пушит в реестр и поднимает полный стек
через [`docker-compose.actions.yml`](docker-compose.actions.yml):
`postgres + redis + migrate + api + worker + beat + mcp`.

В проде миграции выполняет отдельный one-shot сервис `migrate`
(`RUN_MIGRATIONS_ON_STARTUP=false` для api).

---

## Конфигурация

Все настройки — через переменные окружения / `.env.development`
(см. `service/settings.py` и `.env.example`). Ключевое: `DATABASE_URL`,
`REDIS_URL`, `LLM_BACKEND` + модели, `FILESERVER_*` (MinIO), `URBAN_API_BASE_URL`.
Для чат-ручек: `CHAT_STORAGE_BASE_URL` (история диалогов; пусто — персист выключен),
`CHAT_MODEL` (дефолтная модель чата на `OLLAMA_BASE_URL`), `CHAT_SYSTEM_PROMPT_PATH`.
Секреты в репозиторий не коммитятся.
