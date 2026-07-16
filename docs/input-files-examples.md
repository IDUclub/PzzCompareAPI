# Входные файлы — примеры

Справочник по всем входным файлам сервиса: как они выглядят и какой эндпоинт/режим их использует.
Основной контракт эндпоинтов — в [frontend-api-guide.md](frontend-api-guide.md).

## Матрица «эндпоинт → файлы»

| Эндпоинт / режим | Основной слой | Зоны ПЗЗ | Описания зон | Классификатор ВРИ |
|------------------|---------------|----------|--------------|-------------------|
| `POST /tasks/pzz-check` | `cadastral_…` — **участки** (§1) | `pzz_zones_…` (§3) | `pzz_zone_vri_labels_file` (§4.1) | `vri_classifier_file` (§5) |
| `POST /tasks/classify-only` | `cadastral_…` — **участки** (§1) | — | — | `vri_classifier_file` (§5) |
| `POST /scenarios/{id}/classify` | — данные тянутся из urban_api по `scenario_id`, файлы не грузятся | | | |
| `auto/chat/stream` `mode=pzz_check` | `cadastral_…` — **участки** (§1) | `pzz_zones_…` (§3) | `pzz_zone_vri_labels_file` (§4.1) | `vri_classifier_file` (§5) |
| `auto/chat/stream` `mode=classify_only` | `cadastral_…` — **участки** (§1) | — | — | `vri_classifier_file` (§5) |
| `auto/chat/stream` `mode=building_pzz_check` | `cadastral_…` — **здания** (§2) | `pzz_zones_…` (§3) | `pzz_descriptions_file` (§4.2) | — |
| `POST /pzz/zone-descriptions/convert` | `file` — таблица описаний зон CSV/XLSX (§4.3) → JSON §4.2-B | — | — | — |

**Общее для всех гео-файлов:**
- CRS **всегда EPSG:4326**.
- Форматы: `.geojson` / `.json`, либо GeoPackage `.gpkg` / `.gml` / `.kml` / GeoParquet `.parquet`
  (репроецируются в EPSG:4326 на приёме).
- У ручек `/tasks/pzz-check` и `/tasks/classify-only` имена колонок задаются **явно**
  (`cadastral_vri_col`, `pzz_zone_code_col`, `pzz_zone_name_col`). В `auto/chat/stream` колонки
  **определяются автоматически** по содержимому и известным именам.
- Лимит файла — 200 МБ.

---

## 1. Кадастровые участки (`cadastral_feature_collection_file`)

Для `pzz_check` / `classify_only`. Полигоны участков; в одном текстовом поле — ВРИ участка.
Имя поля указывается в `cadastral_vri_col` (или определяется авто; известные имена: `Вид_разрешенного_исп`,
`вид разрешенного использования`, `ври`, …).

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "Polygon", "coordinates": [
        [[31.030,59.940],[31.030,59.942],[31.034,59.942],[31.034,59.940],[31.030,59.940]]
      ]},
      "properties": { "Вид_разрешенного_исп": "Для индивидуального жилищного строительства" }
    },
    {
      "type": "Feature",
      "geometry": { "type": "Polygon", "coordinates": [
        [[31.040,59.950],[31.040,59.952],[31.044,59.952],[31.044,59.950],[31.040,59.950]]
      ]},
      "properties": { "Вид_разрешенного_исп": "Склады" }
    }
  ]
}
```

ВРИ здесь — **текст** (человекочитаемое наименование); бэкенд сам сопоставляет его с кодом ВРИ
из классификатора Росреестра (§5).

---

## 2. Слой зданий (`cadastral_feature_collection_file` в `building_pzz_check`)

Каждая фича — один объект. Тип/сервис — числовым `id` (Urban API), текстом или вложенным
urban_api-объектом; этажность опциональна. Единица проверки: **один объект = одна фича = один
вердикт** (жилой дом и размещённый в нём сервис — ДВЕ отдельные фичи).

```json
{
  "type": "FeatureCollection",
  "features": [
    { "type": "Feature", "geometry": { "type": "Point", "coordinates": [31.031, 59.941] },
      "properties": { "physical_object_type_id": 4, "floors_count": 3 } },

    { "type": "Feature", "geometry": { "type": "Point", "coordinates": [31.032, 59.942] },
      "properties": { "physical_object_type_name": "Жилой дом", "floors_count": 9 } },

    { "type": "Feature", "geometry": { "type": "Point", "coordinates": [31.033, 59.943] },
      "properties": { "service_type_id": 22 } },

    { "type": "Feature", "geometry": { "type": "Point", "coordinates": [31.034, 59.944] },
      "properties": { "service_type_name": "Детский сад" } },

    { "type": "Feature", "geometry": { "type": "Point", "coordinates": [31.035, 59.945] },
      "properties": { "physical_object_type_name": "Склад" } },

    { "type": "Feature", "geometry": { "type": "Point", "coordinates": [31.036, 59.946] },
      "properties": {
        "physical_object_type": { "physical_object_type_id": 4, "name": "Жилой дом" },
        "properties": { "Количество этажей": 5 } } }
  ]
}
```

**Распознаваемые имена колонок здания:**

| Роль | Известные имена | Значение |
|------|-----------------|----------|
| тип | `physical_object_type_id`, `physical_object_type_name`, `physical_object_type` (вложенный), `тип`, `building_type`, `po_type_id` | число (напр. `4` = жилой дом) **или** текст («Жилой дом», «Склад») |
| сервис | `service_type_id`, `service_type_name`, `service_type_code`, `service_type` (вложенный), `сервис`, `service` | число `service_type_id` **или** текст/код («Школа», `school`) |
| этажность | `floors_count`, `floors`, `этажность`, `количество этажей`, `number_of_floors`; вложенно `properties["Количество этажей"]` | целое (`1`, `5`, `24`) |

Нужна колонка типа **или** сервиса; приоритет подбора ВРИ: жилое (по этажности) → `service_type_id`
→ `physical_object_type_id`.

**Тип/сервис можно писать произвольным человеческим текстом** — не только id или каноничным именем
Urban API. Разрешение имени в справочник идёт лестницей:

```
числовой id → точное/префиксное совпадение имени (словарь)      ← без ЛЛМ
            → семантический подбор по смыслу (эмбеддер)          ← «автомойка самообслуживания» → дорожный сервис
            → ЛЛМ (запасной)
            → не разрешилось → «Требуется ручная проверка»       (ВРИ не выдумывается)
```

Как именно подобрано — видно в колонке результата `Основание_подбора_ВРИ`; неточное сопоставление
помечается «(название сопоставлено по смыслу)» (эмбеддер) или «(сопоставлено ИИ)» (ЛЛМ).

> Семантический подбор требует поднятого векторайзера (`VECTORIZER_URL` + `EMBED_MODEL`). Если он
> выключен/недоступен — работает связка «словарь → ЛЛМ → ручная проверка».

---

## 3. Слой зон ПЗЗ (`pzz_zones_feature_collection_file`)

Ключевое — колонка **кода зоны** (`pzz_zone_code_col` / авто; известные имена: `Индекс_зоны`,
`zone_code`, `index`, `индекс`). Название зоны — `zone_name` / `Код_объекта` / `name` / `наименование`.

### Вариант A — числовой `functional_zone_type_id` (urban_api)

```json
{
  "type": "FeatureCollection",
  "features": [
    { "type": "Feature",
      "geometry": { "type": "Polygon", "coordinates": [
        [[31.02,59.94],[31.02,59.96],[31.06,59.96],[31.06,59.94],[31.02,59.94]]
      ]},
      "properties": { "zone_code": 8, "zone_name": "Жилая зона" } }
  ]
}
```

При отсутствии `zone_code` берётся вложенный `functional_zone_type.id`.

### Вариант B — буквенный индекс ПЗЗ («Ж-1», «П-1»…)

Только для `building_pzz_check`. Разрешённые ВРИ берутся по буквенной схеме описаний (§4.2) или из
встроенного шаблона (приблизительно → двухфазный флоу, §6).

```json
{
  "type": "FeatureCollection",
  "features": [
    { "type": "Feature",
      "geometry": { "type": "Polygon", "coordinates": [
        [[31.02,59.94],[31.02,59.96],[31.06,59.96],[31.06,59.94],[31.02,59.94]]
      ]},
      "properties": { "Индекс_зоны": "Ж-1", "zone_name": "Зона застройки ИЖС" } }
  ]
}
```

---

## 4. Описания / лейблы зон

Разные ручки используют разные файлы описаний. **Оба** описывают «зона → разрешённые ВРИ» через
секции `main` / `conditional` / `auxiliary`, но подаются в разные поля.

### 4.1. `pzz_zone_vri_labels_file` — для `pzz_check` (проверка **участков**)

Список зон; ключ `zone_code`. Грунтует LLM-классификацию участков. Полная форма несёт много
метаданных (`article_code`, `zone_heading`, `zone_summary`, …), но минимально достаточно
`zone_code` + `zone_name` + секции с `vri_code`/`vri_name`. Без файла — встроенный
`pzz_zone_llm_labels_template.json`.

```json
[
  {
    "zone_code": "Ж-1",
    "zone_name": "ЗОНА ЗАСТРОЙКИ ИНДИВИДУАЛЬНЫМИ ЖИЛЫМИ ДОМАМИ",
    "main":        [{ "vri_code": "2.1", "vri_name": "Для индивидуального жилищного строительства" }],
    "conditional": [{ "vri_code": "3.5.1", "vri_name": "Дошкольное, начальное и среднее общее образование" }],
    "auxiliary":   []
  }
]
```

### 4.2. `pzz_descriptions_file` — для `building_pzz_check` (проверка **зданий**)

Принимаются **обе** схемы (бэкенд выбирает по содержимому). Ключ зоны должен совпадать с кодом
зоны из §3.

**Схема A — числовая** (под `functional_zone_type_id`):

```json
{
  "functional_zone_mappings": [
    {
      "functional_zone_type_id": 8,
      "db_zone_nickname": "Жилая зона",
      "averaged_pzz_profile": {
        "main_vri":        [{ "vri_code": "2.1" }, { "vri_code": "2.5" }],
        "conditional_vri": [{ "vri_code": "3.5.1" }],
        "auxiliary_vri":   [{ "vri_code": "12.0" }]
      }
    }
  ]
}
```

**Схема B — буквенная** (под индекс ПЗЗ):

```json
[
  { "zone_code": "Ж-1", "zone_name": "Зона застройки ИЖС",
    "main": [{ "vri_code": "2.1" }], "conditional": [{ "vri_code": "3.5.1" }], "auxiliary": [] },
  { "zone_code": "П-1", "zone_name": "Зона производственных объектов",
    "main": [{ "vri_code": "6.9" }], "conditional": [], "auxiliary": [] }
]
```

> Буквенная схема §4.2-B и лейблы §4.1 — это один и тот же формат «список зон по `zone_code` с
> `main/conditional/auxiliary`». Различаются только тем, в какое поле подаются и что классифицируют
> (участки vs здания).

ВРИ проверяется на вхождение в `main` / `conditional` / `auxiliary`, иерархически (зонтичный код
разрешает вложенные: `2.1` покрывает `2.1.1`).

### 4.3. Таблица описаний зон (CSV / XLSX) → `POST /pzz/zone-descriptions/convert`

JSON §4.2 не обязательно готовить руками — описания зон часто уже лежат таблицей (выгрузка
градрегламента). Эндпоинт принимает **CSV или XLSX**, определяет колонки **через ЛЛМ** (модель
смотрит на имена колонок и первые значения и назначает роли — заголовки могут быть любыми) и
**детерминированно** сворачивает в буквенную схему §4.2-B. Задача при этом не создаётся — это шаг
предпросмотра: фронт показывает результат, пользователь проверяет и затем подаёт полученный JSON как
`pzz_descriptions_file`.

**Ожидаемый формат — «tidy», одна строка = одно разрешение:**

| роль (ключ) | смысл | обязательна | пример |
| --- | --- | --- | --- |
| `zone_code` | код (индекс) зоны | **да** | `Ж-1` |
| `zone_name` | наименование зоны | нет (по умолч. = код) | `Зона застройки ИЖС` |
| `permission` | тип разрешения | нет¹ | `Основной` |
| `vri_code` | код ВРИ | **да** | `2.1` |
| `vri_name` | наименование ВРИ | нет | `Для ИЖС` |

Роли назначает ЛЛМ по именам колонок и примерам значений, поэтому заголовки могут быть произвольными
(`Территориальная зона`, `Разрешение`, `Индекс ВРИ`…). Если модель недоступна, срабатывает
детерминированный бэкстоп — точное совпадение имени со словарём (`Zone`→`zone_code`, `Code`→
`permission`, `VRI_Code`→`vri_code`, `VRI`→`vri_name`), так что стандартная выгрузка конвертируется и
офлайн. `permission` затем сворачивается по подстроке: `основн`→`main`, `условн`→`conditional`,
`вспомог`→`auxiliary`. Значение `column_map` переопределяет выбор модели для любой роли.

> ¹ Если колонки `permission` нет — **все** ВРИ уходят в `main`, в `report.warnings` пишется
> предупреждение. Нераспознанное значение `permission` тоже уходит в `main` с warning.

**Кодировка:** XLSX — Unicode нативно. CSV — сниф: `utf-8-sig` (в т.ч. BOM), затем `cp1251`
(типичный русский Excel-CSV с разделителем `;`); разделитель (`;`/`,`/tab) тоже определяется
автоматически. Слепого декодирования нет — кириллица не бьётся.

Пример CSV (`cp1251;`):

```csv
Zone;Code;VRI_Code;VRI
Ж-1;Основной;2.1;Для индивидуального жилищного строительства
Ж-1;Условно разрешённый;4.7;Гостиничное обслуживание
Ж-1;Вспомогательный;4.9;Служебные гаражи
```

Запрос (`sheet` — только для XLSX; `column_map` — опциональное JSON-переопределение автодетекта):

```bash
curl -X POST http://<host>/pzz/zone-descriptions/convert \
  -F "file=@Долинский.xlsx" \
  -F "sheet=pzz_Regl" \
  -F 'column_map={"permission":"Code"}'
```

Ответ:

```json
{
  "zones": [
    { "zone_code": "Ж-1", "zone_name": "Ж-1",
      "main": [{ "vri_code": "2.1", "vri_name": "Для индивидуального жилищного строительства" }],
      "conditional": [{ "vri_code": "4.7", "vri_name": "Гостиничное обслуживание" }],
      "auxiliary": [{ "vri_code": "4.9", "vri_name": "Служебные гаражи" }] }
  ],
  "columns_detected": {
    "zone_code":  { "column": "Zone",     "source": "llm",      "title": "код (индекс) зоны ПЗЗ" },
    "permission": { "column": "Code",     "source": "override", "title": "тип разрешения ВРИ (основной/условный/вспомогательный)" },
    "vri_code":   { "column": "VRI_Code", "source": "llm",      "title": "код ВРИ" }
  },
  "report": { "zones_count": 22, "vri_count": 485, "rows_total": 486, "rows_used": 486, "warnings": [] }
}
```

Массив `zones` — ровно схема §4.2-B; его и подавайте как `pzz_descriptions_file`.
Ошибки: `400` — неподдержанный формат / битая кодировка / пустой файл / нет листа;
`422` — не удалось определить обязательную колонку (`zone_code`/`vri_code`), в `detail` — что нашли и
какие заголовки есть (передайте `column_map`).

> **Инлайн, без `/convert`:** ту же таблицу `.csv`/`.xlsx` можно подать напрямую в
> `POST /tasks/auto/chat/stream` — как `pzz_descriptions_file` (`building_pzz_check`) или как
> `pzz_zone_vri_labels_file` (`pzz_check`): конвертация произойдёт на месте, распознанное (зоны/ВРИ +
> warnings) попадёт в ведущий `chunk` нарратива, а при нераспознанной обязательной колонке придёт
> терминальный `error`. `/convert` нужен, когда маппинг колонок хочется проверить ДО запуска.

---

## 5. Классификатор ВРИ (`vri_classifier_file`, опционально)

Федеральный классификатор Росреестра (справочник кодов ВРИ). По умолчанию встроен
(`rosreestr_vri_classifier_2024_12_24.json`) — **переопределять почти никогда не нужно**. Форма —
объект с `entries` (плоский список кодов) и индексом `by_code`:

```json
{
  "source": { "title": "Классификатор видов разрешённого использования земельных участков",
              "document": "Приказ Росреестра от 10.11.2020 N П/0412" },
  "entries": [
    { "code": "2.1", "name": "Для индивидуального жилищного строительства",
      "description": "Размещение жилого дома …", "parent_code": "2.0", "level": 2 },
    { "code": "6.9", "name": "Склады",
      "description": "Размещение сооружений …", "parent_code": "6.0", "level": 2 }
  ]
}
```

---

## 6. `confirmed_zone_map` (двухфазный флоу building_pzz_check)

Не файл, а **form-поле со строкой JSON** — подтверждённые пользователем соответствия «код юзера →
код зоны шаблона», отправляемые во втором запросе после события `zone_review`/`confirm`:

```json
{ "СХ-3": "АГ-1", "Т-1": "П-1" }
```

Полный порядок двухфазного флоу — [frontend-api-guide.md](frontend-api-guide.md), раздел H
(«Зоны … + двухфазное подтверждение») и «Сценарий 5».

---

## 7. Примеры запросов (multipart)

### `pzz-check` — участки + зоны (колонки заданы явно)

```bash
curl -X POST https://<host>/tasks/pzz-check \
  -F "cadastral_feature_collection_file=@parcels.geojson;type=application/geo+json" \
  -F "pzz_zones_feature_collection_file=@zones.geojson;type=application/geo+json" \
  -F "cadastral_vri_col=Вид_разрешенного_исп" \
  -F "pzz_zone_code_col=Индекс_зоны" \
  -F "pzz_zone_name_col=zone_name"
# → TaskOut { external_id, status:"queued" }; далее поллинг GET /tasks/{external_id}
```

### `classify_only` — только участки

```bash
curl -X POST https://<host>/tasks/classify-only \
  -F "cadastral_feature_collection_file=@parcels.geojson;type=application/geo+json" \
  -F "cadastral_vri_col=Вид_разрешенного_исп"
```

### `auto/chat/stream` — числовые зоны зданий (одна фаза, Bearer обязателен)

```bash
curl -N -X POST https://<host>/tasks/auto/chat/stream \
  -H "Authorization: Bearer <jwt>" \
  -F "mode=building_pzz_check" \
  -F "cadastral_feature_collection_file=@buildings.geojson;type=application/geo+json" \
  -F "pzz_zones_feature_collection_file=@zones_numeric.geojson;type=application/geo+json"
```

### `auto/chat/stream` — буквенные зоны зданий, двухфазно

```bash
# Фаза 1 — без confirmed_zone_map → придёт zone_review (confirm|suggest_upload), задача НЕ создаётся
curl -N -X POST https://<host>/tasks/auto/chat/stream \
  -H "Authorization: Bearer <jwt>" \
  -F "mode=building_pzz_check" \
  -F "cadastral_feature_collection_file=@buildings.geojson;type=application/geo+json" \
  -F "pzz_zones_feature_collection_file=@zones_letters.geojson;type=application/geo+json"

# Фаза 2 — повтор с подтверждёнными парами (confirmed_zone_map — обычное form-поле)
curl -N -X POST https://<host>/tasks/auto/chat/stream \
  -H "Authorization: Bearer <jwt>" \
  -F "mode=building_pzz_check" \
  -F "cadastral_feature_collection_file=@buildings.geojson;type=application/geo+json" \
  -F "pzz_zones_feature_collection_file=@zones_letters.geojson;type=application/geo+json" \
  -F 'confirmed_zone_map={"СХ-3":"АГ-1","Т-1":"П-1"}'
```

Что возвращается (GeoJSON-результат с колонками ПЗЗ, два `file`-слоя для зданий, SSE-события) —
[frontend-api-guide.md](frontend-api-guide.md), разделы D2 и H.
