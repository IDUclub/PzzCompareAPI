# MCP-контракт PzzCompare для взаимодействия агентов

Эта справка описывает, как расширять MCP для PzzCompare под мультиагентные сценарии. Целевой потребитель - другой агент, а не фронтендовый чат.

## Принцип

MCP PzzCompare должен возвращать ответы другим агентам обычным tool response.

Не добавляем:

- историю чата;
- `chat_id`;
- ChatStorage;
- `service_event`;
- SSE-поток;
- проксирование `/chat/stream`.

Добавляем stateless-инструменты, которые возвращают JSON:

- статус выполнения;
- идентификатор задачи;
- структурированный отчет;
- готовое текстовое резюме;
- ссылки или идентификаторы выходных геослоев, если они доступны;
- `next_step`, если агенту нужно запросить действие пользователя.

## Текущая архитектура

MCP-сервер находится в `service/mcp_server/`. Он работает как отдельный процесс FastMCP и ходит в основной FastAPI сервис через `MCP_API_BASE_URL`.

Ключевые файлы:

- `service/mcp_server/main.py` - создает root `FastMCP` и монтирует домены tools;
- `service/mcp_server/dependencies.py` - создает singleton `ApiClient`;
- `service/mcp_server/api_client.py` - тонкий HTTP-клиент к FastAPI;
- `service/mcp_server/tools/tasks.py` - MCP-инструменты для файловых задач;
- `service/mcp_server/tools/scenarios.py` - MCP-инструменты для scenario flow;
- `service/mcp_server/exceptions.py` - перевод REST-ошибок в MCP/JSON-RPC ошибки.

Сейчас уже есть базовая цепочка:

```text
submit_building_pzz_check_task
  -> get_task_status
  -> get_task_report
```

Она закрывает одиночную проверку, но другим агентам удобнее получить результат одним tool-вызовом.

## Tool 1. check_building_pzz_and_wait

Назначение: запустить проверку зданий по ПЗЗ, дождаться результата и вернуть отчет.

Подходит для сценариев:

- комплексное развитие территории;
- предынвестиционная оценка;
- проверка школы или детского сада;
- одиночная проверка после корректировки.

### Вход

```json
{
  "buildings_upload_id": "string",
  "pzz_zones_upload_id": "string",
  "descriptions_upload_id": "string | null",
  "confirmed_zone_map": {
    "user_zone_code": "template_zone_code"
  },
  "group_by": "zone",
  "priority": 1,
  "force_recompute": false,
  "timeout_seconds": 300
}
```

Поля:

- `buildings_upload_id` - обязательный id загруженного слоя зданий;
- `pzz_zones_upload_id` - обязательный id загруженного слоя зон ПЗЗ;
- `descriptions_upload_id` - опциональный id файла описаний ПЗЗ;
- `confirmed_zone_map` - опциональная карта подтвержденных соответствий зон;
- `group_by` - `zone` или `object`, по умолчанию `zone`;
- `priority` - приоритет задачи;
- `force_recompute` - принудительный пересчет;
- `timeout_seconds` - максимум ожидания.

### Успешный выход

```json
{
  "action": "finished",
  "external_id": "task-id",
  "status": "finished",
  "summary": {
    "total": 120,
    "in_correct_zone": 105,
    "in_wrong_zone": 8,
    "unclear": 7,
    "not_in_zone": 2
  },
  "chat_message": "Проверено зданий: ...",
  "report": {
    "task_external_id": "task-id",
    "group_by": "zone",
    "summary": {},
    "zones": []
  },
  "next_step": null
}
```

### Нетерминальный выход

Если задача не успела завершиться:

```json
{
  "action": "timeout",
  "external_id": "task-id",
  "status": "running",
  "ready": false,
  "next_step": "Call get_task_status until status is finished, then call get_task_report."
}
```

### Выход, требующий решения пользователя

Если `submit_building_pzz_check_task` вернул `confirm` или `suggest_upload`, tool должен вернуть это как результат, а не повторять вызов:

```json
{
  "action": "confirm",
  "status": "blocked_by_user_input",
  "chat_message": "Часть зон не найдена в шаблоне ПЗЗ...",
  "suggestions": [
    {
      "user_code": "Ж-1",
      "user_name": "Жилая зона",
      "suggested_code": "Ж-1",
      "suggested_name": "Зона жилой застройки"
    }
  ],
  "next_step": "Show the suggested zone matches to the user and resubmit with confirmed_zone_map."
}
```

Правило: MCP не должен сам подтверждать `confirmed_zone_map`. Это должен сделать пользователь или агент-координатор по явному решению.

### Реализация

Файл: `service/mcp_server/tools/tasks.py`.

Алгоритм:

1. Вызвать `api.submit_building_pzz_check(...)`.
2. Если `action != "created"`, вернуть ответ API как есть, добавив `status: "blocked_by_user_input"` при необходимости.
3. Если `action == "created"`, взять `external_id` из `task`.
4. Поллить `api.get_task(external_id)` до `finished`, `failed` или таймаута.
5. Во время ожидания вызывать `ctx.report_progress(...)`.
6. При `finished` вызвать `api.get_task_report(external_id, group_by)`.
7. Вернуть нормализованный JSON.

## Tool 2. compare_pzz_reports

Назначение: сравнить ПЗЗ-результаты нескольких вариантов мастер-плана.

Подходит для сценария сравнения вариантов.

### Вход

```json
{
  "variants": [
    {
      "label": "Вариант А",
      "external_id": "task-a"
    },
    {
      "label": "Вариант Б",
      "external_id": "task-b"
    },
    {
      "label": "Вариант В",
      "external_id": "task-c"
    }
  ],
  "group_by": "zone"
}
```

### Выход

```json
{
  "items": [
    {
      "label": "Вариант А",
      "external_id": "task-a",
      "status": "finished",
      "total": 100,
      "in_correct_zone": 90,
      "in_wrong_zone": 6,
      "unclear": 4,
      "not_in_zone": 1
    }
  ],
  "best_by_pzz": null,
  "chat_message": "PzzCompare не выбирает лучший вариант без заданных критериев и весов. По ПЗЗ меньше всего потенциальных несоответствий у варианта ..."
}
```

Правило: tool не должен самостоятельно выбирать лучший мастер-план. Он может подсветить минимум/максимум по отдельным метрикам, но итоговый выбор остается за агентом-координатором или пользователем.

### Реализация

Файл: `service/mcp_server/tools/tasks.py`.

Алгоритм:

1. Для каждого `external_id` вызвать `api.get_task(external_id)`.
2. Если задача не `finished`, вернуть по варианту `ready: false`, `status`, `next_step`.
3. Для finished-задач вызвать `api.get_task_report(external_id, group_by)`.
4. Из `report.summary` собрать таблицу:
   - `total`;
   - `in_correct_zone`;
   - `in_wrong_zone`;
   - `unclear`;
   - `not_in_zone`.
5. Сформировать короткий `chat_message` для другого агента.

## Tool 3. compare_pzz_versions

Назначение: сравнить ПЗЗ-результат до и после корректировки проектного решения.

Подходит для сценария доработки после замечаний.

### Вход

```json
{
  "before_external_id": "task-before",
  "after_external_id": "task-after",
  "group_by": "zone"
}
```

### Выход

```json
{
  "before": {
    "external_id": "task-before",
    "summary": {
      "total": 120,
      "in_wrong_zone": 8,
      "unclear": 7,
      "not_in_zone": 2
    }
  },
  "after": {
    "external_id": "task-after",
    "summary": {
      "total": 118,
      "in_wrong_zone": 5,
      "unclear": 4,
      "not_in_zone": 2
    }
  },
  "delta": {
    "total": -2,
    "in_wrong_zone": -3,
    "unclear": -3,
    "not_in_zone": 0
  },
  "chat_message": "После корректировки потенциальных несоответствий ПЗЗ стало меньше: 8 -> 5. Объектов на ручной проверке стало меньше: 7 -> 4."
}
```

### Ограничение

Сейчас отчет `object-zone-fit` надежно содержит `feature_index`, но не гарантирует стабильный идентификатор исходного объекта. Поэтому tool может корректно сравнить агрегированные метрики, но не всегда может доказательно сказать, какой именно объект исправился.

Для точного object-level diff нужно добавить в результат и отчет устойчивый исходный id объекта, например:

- `object_id`;
- `building_id`;
- `service_id`;
- `source_feature_id`;
- или другой идентификатор из Urban API/входного слоя.

## Нормализованная модель ответа

Для всех новых MCP tools рекомендуется использовать единый верхнеуровневый формат:

```json
{
  "action": "finished | timeout | failed | confirm | suggest_upload | detection_failed",
  "ready": true,
  "status": "finished",
  "external_id": "task-id",
  "summary": {},
  "report": {},
  "chat_message": "string",
  "warnings": [],
  "next_step": null
}
```

Смысл полей:

- `action` - что произошло с точки зрения агента;
- `ready` - можно ли использовать результат дальше;
- `status` - статус task/job;
- `external_id` - id задачи PzzCompare;
- `summary` - компактные метрики;
- `report` - полный структурированный отчет;
- `chat_message` - готовый человекочитаемый текст для агента-координатора;
- `warnings` - нефатальные ограничения;
- `next_step` - что должен сделать следующий агент или пользователь.

## Ошибки

Сохраняем текущую схему `map_errors`:

- `-32602` - некорректные параметры или upstream 4xx;
- `-32603` - внутренняя ошибка или upstream 5xx;
- `-32002 AUTH_TOKEN_EXPIRED` - пользовательский Bearer был отклонен.

Для состояний `confirm`, `suggest_upload`, `detection_failed` лучше возвращать обычный JSON, а не MCP error: это не техническая ошибка, а нормальный шаг процесса.

## Тесты

Минимальный набор unit-тестов:

- `check_building_pzz_and_wait` возвращает `confirm` без polling;
- `check_building_pzz_and_wait` возвращает `suggest_upload` без polling;
- `check_building_pzz_and_wait` поллит до `finished` и затем вызывает `get_task_report`;
- `check_building_pzz_and_wait` возвращает `timeout`, если задача не завершилась;
- `compare_pzz_reports` собирает таблицу по нескольким finished-задачам;
- `compare_pzz_reports` корректно показывает unfinished-вариант;
- `compare_pzz_versions` считает delta по `summary`;
- `compare_pzz_versions` сообщает ограничение object-level diff без устойчивого id.

Подходящий файл для тестов:

- `tests/test_mcp_pzz_agent_tools.py`.

Существующий пример тестового стиля:

- `tests/test_mcp_integrator_flows.py`.

## Короткая формулировка для коллеги

MCP PzzCompare - это не чат и не UI-стрим. Это stateless API инструментов для других агентов. История, `chat_id`, ChatStorage и SSE остаются во фронтовых `/chat/stream` ручках. Для мультиагентного взаимодействия MCP должен возвращать готовые структурированные ответы: одиночный ПЗЗ-отчет, сравнение вариантов и сравнение версий до/после.
