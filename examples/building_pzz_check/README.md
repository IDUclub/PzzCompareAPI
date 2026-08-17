# Тестовые файлы — `building_pzz_check`

Синтетический, но самосогласованный комплект для ручной проверки режима `building_pzz_check`
(`POST /tasks/auto/chat/stream`, `mode=building_pzz_check`). Все точки зданий гарантированно
попадают внутрь полигонов зон (проверено shapely). CRS — EPSG:4326.

Три зоны рядом: **Ж-1** (жилая), **П-1** (производственная), **Р-1** (рекреация); в каждой — по три
здания, покрывающие все ветки подбора ВРИ.

## Файлы

| Файл | Что это | §  |
|------|---------|----|
| `buildings.geojson` | 9 зданий-точек: id / текст / вложенный urban_api-объект, разная этажность | §2 |
| `zones_letters.geojson` | зоны с буквенным индексом (`Ж-1`/`П-1`/`Р-1`) + `zone_name` | §3-B |
| `zones_numeric.geojson` | те же полигоны с числовым `zone_code` (8/6/5) + `zone_name` | §3-A |
| `pzz_descriptions_letters.json` | описания под буквенные зоны (схема B) | §4.2-B |
| `pzz_descriptions_numeric.json` | описания под числовые зоны (`functional_zone_mappings`) | §4.2-A |
| `pzz_descriptions_table.csv` | та же таблица описаний, **cp1251 + `;`** (русский Excel-CSV) | §4.3 |
| `pzz_descriptions_table_utf8.csv` | та же таблица, **utf-8-sig + `,`** (проверка снифа кодировки) | §4.3 |
| `pzz_descriptions_table.xlsx` | та же таблица, лист `pzz_Regl` | §4.3 |

Таблицы (`*.csv`/`.xlsx`) намеренно с «человеческими» заголовками (`Территориальная зона`,
`Разрешение`, `Индекс ВРИ`…), чтобы проверять определение колонок по значениям, а не по именам.
Все три конвертируются в те же 3 зоны / 12 ВРИ, что и `pzz_descriptions_letters.json`.

## Что проверяет набор зданий

| # | Зона | Свойства | Ветка подбора |
|---|------|----------|---------------|
| 0 | Ж-1 | `physical_object_type_id=4`, 5 эт. | жилое по этажности → в зоне |
| 1 | Ж-1 | `service_type_name="Детский сад"` | сервис → условно разрешён |
| 2 | Ж-1 | `physical_object_type_name="Склад"` | тип текстом → **нарушение** (склад в жилой) |
| 3 | П-1 | `physical_object_type_id=22`, 1 эт. | тип по id → в зоне |
| 4 | П-1 | вложенный `physical_object_type`, 9 эт. | жилой дом в производственной → **нарушение** |
| 5 | П-1 | `service_type_name="автомойка самообслуживания"` | **семантический подбор** эмбеддером |
| 6 | Р-1 | `physical_object_type_id=4`, 2 эт. | жилой дом в рекреации → **нарушение** |
| 7 | Р-1 | `service_type_name="кафе"` | сервис → общественное питание |
| 8 | Р-1 | `physical_object_type_name="антикафе"` | экзотика → ожидаемо **ручная проверка** |

Здание 5 требует поднятого векторайзера (`VECTORIZER_URL` + `EMBED_MODEL`); без него уйдёт в
ЛЛМ-фолбэк или в ручную проверку.

## Запуск

Числовые зоны — одна фаза (задача создаётся сразу):

```bash
curl -N -X POST http://localhost:8000/tasks/auto/chat/stream \
  -H "Authorization: Bearer <jwt>" \
  -F "mode=building_pzz_check" \
  -F "cadastral_feature_collection_file=@buildings.geojson;type=application/geo+json" \
  -F "pzz_zones_feature_collection_file=@zones_numeric.geojson;type=application/geo+json" \
  -F "pzz_descriptions_file=@pzz_descriptions_numeric.json;type=application/json"
```

Буквенные зоны + описания пользователя (собственный ПЗЗ → `proceed` без двухфазного подтверждения):

```bash
curl -N -X POST http://localhost:8000/tasks/auto/chat/stream \
  -H "Authorization: Bearer <jwt>" \
  -F "mode=building_pzz_check" \
  -F "cadastral_feature_collection_file=@buildings.geojson;type=application/geo+json" \
  -F "pzz_zones_feature_collection_file=@zones_letters.geojson;type=application/geo+json" \
  -F "pzz_descriptions_file=@pzz_descriptions_letters.json;type=application/json"
```

Описания **таблицей** (конвертация инлайн; замените json на csv/xlsx):

```bash
  -F "pzz_descriptions_file=@pzz_descriptions_table.csv;type=text/csv"
```

Только предпросмотр конвертации таблицы (задача не создаётся, авторизация не нужна):

```bash
curl -X POST http://localhost:8000/pzz/zone-descriptions/convert \
  -F "file=@pzz_descriptions_table.xlsx" \
  -F "sheet=pzz_Regl"
```

Двухфазный флоу (буквенные зоны **без** описания → обобщённый шаблон, `zone_review`) — просто
уберите `pzz_descriptions_file` из запроса с `zones_letters.geojson`.
