# Инструкция: протестировать интеграцию ChatStorage + чат-ответ в PzzCompareAPI

> Самодостаточная инструкция для агента, который стартует «с нуля» без контекста разработки.

## Контекст (что тестируем)

В сервис `PzzCompareAPI` добавлены: клиент **ChatStorage**, стриминговый клиент **Ollama
`/api/chat`**, разговорный слой поверх классификации и приём gdf-форматов на загрузку. Две новые
ручки (SSE):

- `POST /scenarios/{id}/chat/stream` — классификация по urban_api → стрим ответа LLM.
- `POST /tasks/pzz-check/chat/stream` — проверка ПЗЗ по загруженным файлам → стрим ответа LLM.
- `POST /tasks/classify-only/chat/stream` — только классификация ВРИ по загруженным файлам → стрим ответа LLM.

Ключевые файлы: `service/infrastructure/{chat_storage_client,ollama_chat_client,geo_ingest}.py`,
`service/application/use_cases/chat_answer.py`, `service/api/{tasks,scenarios,classifier,security}.py`,
`service/settings.py`.

## 0. Окружение

- Python из `./.venv/Scripts/python.exe` (Windows, Git Bash доступен).
- В `.env.development` для функционального теста должны быть заданы:
  `CHAT_STORAGE_BASE_URL=http://10.32.1.99:8010`, `CHAT_MODEL=<модель, которая есть на OLLAMA_BASE_URL>`,
  плюс существующие `OLLAMA_BASE_URL`, `URBAN_API_BASE_URL`, `DATABASE_URL`, `REDIS_URL`.

## 1. Юнит-тесты (должны быть зелёными)

```bash
./.venv/Scripts/python.exe -m pytest \
  tests/test_chat_answer.py tests/test_chat_sse.py tests/test_geo_ingest.py \
  tests/test_chat_storage_client.py tests/test_ollama_chat_client.py -p no:warnings
```

Ожидание: **25 passed**. Покрывает gMART-формат событий (`chunk`/`service_event`), клиентов,
geo-конверсию, загрузку истории чата.

## 2. Известные НЕ-наши падения (игнорировать)

- Полный `pytest tests/` падает на сборке `test_task_result_endpoint.py` + `test_tasks_list_endpoint.py`
  (ошибка парсинга `DATABASE_URL`) — пред-существующая флейка изоляции. По отдельности проходят.
- `tests/test_pipeline_runners.py`, `tests/test_pipeline_runtime_context.py` падают
  (`requests.MissingSchema`) — требуют реальных пайплайновых URL (`VECTORIZER_URL` и т.п.).
  Тоже пред-существующее, не связано с чатом.

Проверка, что это не регресс: `git stash && pytest <файл> && git stash pop` — падения
воспроизводятся и без изменений.

## 3. Sanity импорта/маршрутов

```bash
./.venv/Scripts/python.exe -c "import service.api.scenarios as s, service.api.classifier as c; \
print([r.path for r in s.router.routes if 'chat' in r.path], [r.path for r in c.router.routes if 'chat' in r.path])"
```

Ожидание: `['/scenarios/{scenario_id}/chat/stream'] ['/tasks/pzz-check/chat/stream', '/tasks/classify-only/chat/stream']`.

## 4. Функциональный тест чата (нужны живые сервисы)

Подними API (`docker compose -f docker-compose.yml up -d --build` или `uvicorn`). Нужны: достижимый
ChatStorage, Ollama с `CHAT_MODEL`, urban_api, валидный JWT (`$TOKEN`) и существующий
`scenario_id`/`year`/`source`.

### 4a. Сценарный чат (новый чат)

```bash
curl -N -X POST "http://localhost:8000/scenarios/<SCENARIO_ID>/chat/stream" \
  -H "Authorization: Bearer $TOKEN" \
  -F "user_query=Какие объекты не в своей зоне?" \
  -F "year=<YEAR>" -F "source=<SOURCE>"
```

Ожидаемая последовательность SSE: `task` → `task_event`*/`status`* → `object_zone_fit` →
`service_event` (внутри `storage_event_type:"chat_created"`, запомни `chat_id`) → `chunk`*
(`content.text` дописывается) → финальный `chunk` с `done:true` → `done`.

### 4b. Продолжение диалога (история)

Повтори запрос с `-F "chat_id=<сохранённый chat_id>"`. Событие `service_event` НЕ должно прийти;
в логах сервиса видно загрузку истории; ответ модели должен учитывать прошлый контекст.

### 4c. Аплоуд-чат + gdf

```bash
curl -N -X POST "http://localhost:8000/tasks/pzz-check/chat/stream" \
  -H "Authorization: Bearer $TOKEN" \
  -F "cadastral_feature_collection_file=@parcels.gpkg" \
  -F "pzz_zones_feature_collection_file=@zones.gpkg" \
  -F "user_query=Что не так с участками?" \
  -F "cadastral_vri_col=<col>" -F "pzz_zone_code_col=<col>" -F "pzz_zone_name_col=<col>"
```

Принимаются: `.geojson/.json/.gpkg/.gml/.kml/.geoparquet/.parquet`. Неподдерживаемое расширение →
`415`, битый файл → `400`.

### 4d. Граничные случаи

- Без `Authorization` → `401/403`.
- Если `CHAT_STORAGE_BASE_URL` пуст → ответ всё равно стримится (`chunk`), но `service_event` нет
  и история не пишется.
- Сбой ChatStorage/Ollama → событие `error` (`{message,stage}`), поток не рвётся, в конце `done`.

### 4e. Проверка персиста в ChatStorage

После 4a дёрни напрямую ChatStorage и убедись, что чат и 2 сообщения (`role:user` = `user_query`,
`role:assistant` = полный ответ) сохранены. Сообщение ассистента должно содержать `parts`:
`kind:"text"` (ответ) + `kind:"file"` (ссылка на геослой, `payload.url = /files/result/<id>`).

```bash
curl "http://10.32.1.99:8010/api/v1/chat_history/<chat_id>" -H "Authorization: Bearer $TOKEN"
```

### 4f. Геослой-ссылка (событие `file` + `/files/result`)

В чат-стриме (и в обычных `*/classify/stream`, `*/pzz-check/stream`) должно прийти событие `file`:
```json
{ "type": "file", "content": { "name": "classified_result",
  "url": "…/files/result/<external_id>", "download_url": "<presigned|null>",
  "filename": "<external_id>.geojson", "mime_type": "application/geo+json" } }
```
Поле `role` различает `result` и `input`. В **upload-флоу** дополнительно приходят входные слои
(`role:"input"`, `name:"input_cadastral"/"input_zones"`) — сразу в начале стрима. В сценарном их нет.

Проверь долговечные ссылки (307 → presigned MinIO → GeoJSON), `slot` ∈ result/cadastral/zones:
```bash
curl -IL "http://localhost:8000/files/result/<external_id>"      # 307 → 200 от MinIO
curl -IL "http://localhost:8000/files/cadastral/<external_id>"   # входной кадастр (upload-флоу)
```
`download_url` (если не null) должен качать файл напрямую из MinIO без авторизации.
В ChatStorage сохраняется только `result`-ссылка (как `file`-часть assistant-сообщения); входные слои — нет.

## 5. Что считать успехом

- П.1 = 25 passed; п.3 — маршруты на месте.
- П.4a даёт корректную последовательность SSE-событий в gMART-формате и `chat_id`.
- П.4e: в ChatStorage лежат user+assistant сообщения.
- П.4b: история подхватывается; п.4c: gdf-форматы принимаются.

## 6. Формат SSE-событий чат-части (gMART-конверт `{type, content}`)

```jsonc
// создан новый чат
{ "type": "service_event", "content": { "event_type": "storage_event",
  "event": { "storage_event_type": "chat_created", "chat_id": "...", "chat_title": "..." } } }
// дельта ответа (накапливать content.text)
{ "type": "chunk", "content": { "text": "...", "done": false } }
// конец ответа
{ "type": "chunk", "content": { "text": "", "done": true } }
// нефатальная ошибка LLM/ChatStorage
{ "type": "error", "content": { "message": "...", "stage": "llm|create_chat|load_history|..." } }
```
