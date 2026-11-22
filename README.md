**English** | [Русский](README.ru.md)

# PZZ Pipeline Service

A service that classifies cadastral objects against land use and development
rules (**PZZ**, zoning regulations). It compares a project's functional zones
with the objects located in them, reports which objects are **in the wrong zone**,
and gives a human-readable explanation for the chatbot.

It is built around an existing classification pipeline (`pipeline_modules/`),
wrapped into a production service: an HTTP API, a background worker, object
storage and an MCP server for AI agents.

---

## What it does

Three use cases (matching the items of the technical specification):

1. **Comparing zones with objects** — find out which objects stand outside
   their proper functional zone (currently focused on residential objects).
2. **Zone reference** — for a project, return the list of zones with a text
   description of "what may / may not be built".
3. **Chatbot answer** — ready-made plain text listing the misplaced objects.

Two ways to supply data:

- **Scenario flow** (for agents): data is pulled from IDU `urban_api` by
  `scenario_id` + `(year, source)`. The user's token is forwarded to urban_api.
- **File flow**: the user uploads GeoJSON directly
  (`/tasks/pzz-check`, `/tasks/classify-only`).

Separately, there is a **point check of a VRI (permitted use type) against the
territory's PZZ** for the compliance-checking agent: is a given use permitted in
a zone under the actual PZZ of the project's territory (read via NormGraph),
including the maximum number of floors and height
(see [Checking VRI against the territory's PZZ](#checking-vri-against-the-territorys-pzz)).
The scenario and file flows do not use it.

---

## Architecture

```
                ┌─────────────┐      ┌──────────────┐
   agent ─MCP──▶│  mcp_server │─REST▶│   FastAPI    │
                └─────────────┘      │   (api)      │
   front ─REST───────────────────────▶              │
                                     └──────┬───────┘
                                            │ enqueue (Redis)
                                     ┌──────▼───────┐   ┌──────────┐
                                     │ Celery worker│──▶│ pipeline │
                                     │   + beat     │   │ _modules │
                                     └──────┬───────┘   └──────────┘
                                            │
                          Postgres ◀────────┴────────▶ MinIO (inputs/outputs)
```

Layers inside `service/`:

| Directory | Purpose |
|---|---|
| `api/` | HTTP endpoints (classifier, scenarios, tasks, system) |
| `application/` | use cases (create/start/finish task) — orchestration |
| `domain/` | business rules, ports (repository contracts), task states |
| `infrastructure/` | implementations: DB repositories, MinIO, urban_api client, PZZ mapping, **pipeline runner** |
| `mcp_server/` | MCP server for agents (separate process, thin client over the API) |

`pipeline_modules/` is a **third-party** classification **pipeline** (geopandas +
NLP + LLM). The service calls it through the `PipelineRunner` port
(`infrastructure/runners/pipeline_runner.py`), in-process or in a subprocess,
transparently loading inputs from and uploading results to MinIO.

---

## Technologies

- Python 3.11, uv (dependencies in `pyproject.toml` + `uv.lock`), FastAPI, Celery, SQLAlchemy + Alembic
- PostgreSQL, Redis, MinIO (S3-compatible storage)
- FastMCP 3.x (MCP server), Prometheus metrics
- Docker / Docker Compose

---

## Quick start (local)

You need a filled-in `.env.development` (see `.env.example`) with access to the DB,
Redis, MinIO, the LLM backend and (for the scenario flow) urban_api.

```bash
docker compose -f docker-compose.yml up -d --build
```

This starts 6 services: `postgres`, `redis`, `api` (runs migrations on startup),
`worker`, `beat`, `mcp`.

| Service | Address |
|---|---|
| API | http://localhost:8000 (OpenAPI: `/docs`) |
| MCP | http://localhost:8765/mcp (health: `/health`) |
| Worker metrics | http://localhost:9100/metrics |
| Postgres / Redis | 5432 / 6379 |

Shut down: `docker compose -f docker-compose.yml down`.

---

## Endpoints (main)

**File flow**
- `POST /uploads` — upload a file once and then refer to it by `upload_id` (requires Bearer, lives 24 h); the `url` in the response is a durable link for the chat history
- `GET /uploads/{upload_id}` — download your own upload (requires Bearer; someone else's → 403)

The uploads directory is mounted as a volume (`/app/uploads`) so that `upload_id` survives a redeploy:
otherwise a rollout in the middle of a conversation would invalidate files the user has just attached.
Expired uploads are removed by celery beat (`cleanup-expired-uploads`, interval
`UPLOADS_CLEANUP_INTERVAL_SECONDS`, lifetime `UPLOADS_MAX_AGE_HOURS`).
- `POST /tasks/pzz-check` — full PZZ check (cadastre + zones)
- `POST /tasks/classify-only` — VRI classification against the classifier only
- `POST /tasks/building-pzz-check` — PZZ check of buildings; columns are detected
  automatically. A task is not always created: if the zone codes are not covered by the
  built-in template, the response contains `action` = `confirm` (confirm the proposed
  mappings and retry with `confirmed_zone_map`) or `suggest_upload` (upload your own
  description of permitted VRIs)
- `POST /tasks/auto/chat/stream` with `mode=placement` — site selection; the optional `existing_objects_feature_collection_file` supplies actual objects for `A*`/`B*` separately from the cadastre (`existing_objects_type_col` — explicit type column)
- `POST /tasks/pzz-check/chat/stream` — PZZ check + streamed conversational LLM answer (SSE, requires Bearer)
- `POST /tasks/classify-only/chat/stream` — VRI classification + streamed conversational LLM answer (SSE, requires Bearer)
- `GET /tasks/{id}` · `GET /tasks_list` · `GET /tasks/{id}/result`
- `GET /files/{slot}/{id}` — durable link to a geo layer (`slot`: `result`/`cadastral`/`zones` — the API streams the bytes from MinIO through itself; `result_buildings`/`result_services`/`functional_zones` — assembled on the fly)
- `GET /tasks/{id}/object-zone-fit?group_by=zone|object` — structured report + `chat_message`
- `GET /tasks/{id}/events` · `DELETE /tasks/{id}` · `POST /tasks/{id}/recompute`

Cadastre/zone uploads accept GeoJSON, as well as GeoPackage `.gpkg`, GML, KML,
GeoParquet and a ZIP archive with a Shapefile (`.shp`) or MapInfo (`.tab` / `.mif`) layer;
non-GeoJSON formats are converted to GeoJSON (EPSG:4326) on input.
Each layer is passed either as a file in the request body or as a `<layer>_upload_id` field
with an identifier from `POST /uploads`; the latter is needed by MCP clients where a layer
does not fit into a tool argument.

**Scenario flow** (requires `Authorization: Bearer <jwt>`)
- `POST /scenarios/{id}/classify` — run on urban_api data
- `POST /scenarios/{id}/chat/stream` — run + streamed conversational LLM answer (SSE)
- `POST /scenarios/{id}/chat/stream` with `mode=placement`, `service_name`, `year`, `source` — site selection based on the functional zones and all objects of the scenario, without uploading files
- `POST /scenarios/{id}/placement` — the same selection without the chat stream; `GET /scenarios/{id}/tasks/{external_id}/placement-summary` — summary
- `GET /scenarios/{id}/zones-info` — zones + "what may be built" reference
- `GET /scenarios/{id}/tasks/{external_id}` (+ `/result`, `/object-zone-fit`, `/events`)
- `DELETE` / `POST .../recompute`

**Checking VRI against the territory's PZZ** (requires `Authorization: Bearer <jwt>`)
- `POST /pzz/regulations/check-vri` — is a use permitted in a zone under the territory's
  PZZ; see [Checking VRI against the territory's PZZ](#checking-vri-against-the-territorys-pzz)

**Admin: runtime config** (header `X-Admin-Token: <ADMIN_API_TOKEN>`; empty → 503)
- `GET /admin/config/settings` — current effective settings (secrets masked)
- `GET /admin/config/overrides` — active overrides
- `GET /admin/config/{key}` — effective value + override status
- `PUT /admin/config/{key}` `{value, updated_by?}` — set an override (validated; credential/boot-only/unknown keys → 400)
- `DELETE /admin/config/{key}` — remove an override (restore the deployed value)
- `POST /admin/config/reload` — force a resync of overrides in this process

  Overrides are stored in the `config_override` table (shared by all containers) and synced into
  each process's `os.environ` (TTL ~5 s), also resetting the `Settings` cache — they change
  **without a redeploy** across the whole fleet, including the pipeline subprocess. Secrets
  (`ENV_SECRET`) and boot-only keys cannot be changed. `ADMIN_API_TOKEN` lives in `ENV_SECRET`.

The `*/chat/stream` chat endpoints wait for the classification to finish, then stream (in gMART
format: an envelope `{type, content}`) the report (`object_zone_fit` for pzz-check,
`classify_summary` for classify-only) → `service_event/chat_created` (if no `chat_id` was passed)
→ `chunk`* → `done`, and save the conversation (user + assistant) to **ChatStorage**.
The conversational answer is generated by the same LLM backend as the pipeline (`LLM_BACKEND`):
with `vllm`, the OpenAI-compatible `/v1/chat/completions` at `VLLM_BASE_URL`; otherwise Ollama
`/api/chat` at `OLLAMA_BASE_URL`. The model is the `model` request parameter (default
`CHAT_MODEL`/`GENERATE_MODEL`).

A large GeoJSON result in the chat stream is delivered **as a link** (a `file` event), not inline:
the durable `url = /files/result/{id}` (the API proxies the bytes from MinIO, the link does not expire)
is stored in ChatStorage as the `file` part of the message. `download_url` is always `null`: MinIO is
not reachable from the external network. `url` is always relative: the frontend prepends the
service base URL itself based on `source_service`.

**System**: `GET /health`, `GET /readiness`, `GET /metrics`

Detailed frontend guide (in Russian): [`docs/frontend-api-guide.md`](docs/frontend-api-guide.md).

---

## MCP (for AI agents)

The MCP server (`service/mcp_server/`) is a separate FastMCP process, a thin
client on top of the REST API. It lets an AI agent (Claude and others) run
classification and read reports as a set of tools, without knowing the
service internals. Transport: streamable HTTP at `:8765/mcp` (`MCP_API_BASE_URL`
points to the api).

### Tools

**Scenario tools** (`tools/scenarios.py`):

| Tool | Purpose |
|---|---|
| `classify_scenario` | Start classification on urban_api data → `external_id` |
| `classify_scenario_and_wait` | Same + wait with progress, returns the report directly |
| `get_scenario_classification_status` | Task status (`queued/running/finished/failed`) |
| `get_scenario_classification_report` | "Who is in the wrong zone" report + ready-made `chat_message` |
| `get_scenario_zones_info` | Project zones + "what may be built" reference |
| `recompute_scenario_classification` | Forced recomputation |

**File tools** (`tools/tasks.py`): `submit_pzz_check_task`,
`submit_classify_only_task`, `get_task_status`, `list_tasks`,
`get_task_events`, `get_task_result`, `cancel_task`, `recompute_task`.

**Scenario site selection** (`tools/scenarios.py`): `place_service_in_scenario` →
`get_scenario_classification_status` → `get_scenario_placement_summary`.
It uses a reference mapping of functional zones and NormGraph standards;
the geometry of scenario objects does not replace the actual cadastral and PZZ boundaries.

**Territory PZZ** (`tools/regulations.py`): `check_vri_in_pzz` — point check of a
permitted use against the PZZ of the project's territory (`POST /pzz/regulations/check-vri`).

### Parameters

The agent passes all substantive parameters **explicitly**: `scenario_id`, `year`,
`source`, `physical_object_type_id` (4 = residential), `group_by`, `external_id`
and so on. `scenario_id` specifies which project/scenario to work with.

The only thing that is **not** a tool argument is authentication: the user's Bearer
token is taken from the `Authorization` header and forwarded to urban_api.

### Typical conversation

```
get_scenario_zones_info(scenario_id, year, source)          # reference + access/token check
classify_scenario(scenario_id, year, source, type_id) → external_id
poll get_scenario_classification_status(scenario_id, external_id) → finished
get_scenario_classification_report(scenario_id, external_id) # return chat_message to the user
```

### Token handling

If urban_api rejects the token, the tool returns
`-32002 AUTH_TOKEN_EXPIRED`; the agent must request a fresh Bearer token from the
frontend/user and must **not** retry with the old one.

### Connecting (Claude Desktop)

Via the `mcp-remote` bridge (stdio → HTTP):

```json
{ "command": "cmd", "args": ["/c", "npx", "-y", "mcp-remote", "http://localhost:8765/mcp"] }
```

End-to-end smoke test: `scripts/test_mcp_scenario.py` (requires `URBAN_API_TOKEN`,
optionally `SCENARIO_ID` / `SCENARIO_YEAR` / `SCENARIO_SOURCE`).

Additional references for multi-agent trials (in Russian):
[`docs/pzz-multiagent-scenarios-reference.md`](docs/pzz-multiagent-scenarios-reference.md) —
the role of PzzCompare in the scenarios; [`docs/pzz-mcp-agent-contract.md`](docs/pzz-mcp-agent-contract.md) —
the stateless MCP contract for agent interaction.

---

## Authentication

Scenario endpoints (`/scenarios/*`) require `Authorization: Bearer <jwt>`.
The token is forwarded to urban_api and, optionally, **verified locally**
following the IDUclub pattern (Keycloak JWT via JWKS, see `service/auth/`):

- `AUTH_VERIFY=false` (default) — the token is accepted without signature verification
  (dev, or when an upstream gateway has already verified it);
- `AUTH_VERIFY=true` + `AUTH_SERVER_URL=https://<host>/realms/<realm>` —
  the signature (RS256 via JWKS), issuer and (optionally) audience are verified; a rejected
  token → `401`.

Parameters: `AUTH_SERVER_URL`, `AUTH_CLIENT_ID`, `AUTH_VERIFY_AUD`,
`AUTH_VALID_AUDIENCES`, caches `AUTH_JWKS_CACHE_TTL` / `AUTH_USER_CACHE_TTL`.

### Service token for chat history

The chat endpoints accept the user's token from the frontend, but **the backend saves the chat
history with its own service token** (Keycloak `client_credentials` via
[`idu-service-auth`](https://github.com/IDUclub/idu-service-auth)), not with the user's
token. The reason: the user's token can expire during a long computation, and by the time
the history is written ChatStorage would reject it. The backend extracts `user_id` (`sub`)
from the user's token at the start of the request (while it is still fresh) and passes it
to ChatStorage in the `X-User-Id` header, so the history stays bound to the real user.

Parameters: `KEYCLOAK_URL` (base, without `/realms`), `KEYCLOAK_REALM`,
`KEYCLOAK_CLIENT_ID`, `KEYCLOAK_CLIENT_SECRET`, optionally `KEYCLOAK_SCOPE`. If they
are not set, no service token is created and history persistence is disabled (the answer is
still streamed). The token client is created once per process in the lifespan
(`service/app.py`), with background token refresh.

## Metrics

Task metrics (`queue_wait_seconds`, `task_run_seconds`, `task_fail_total`,
`task_retry_total`) are recorded in the worker process, so they are exposed **by the worker** on
`:9100`, a separate target from the API's `:8000/metrics`. Prometheus must scrape
both.

---

## Tests

```bash
uv sync          # dependencies from uv.lock, including the dev group (pytest, black, pre-commit)
uv run pytest
uv run pre-commit install   # black on every commit
```

Dependencies are changed via `uv add` / `uv remove` (or by editing `pyproject.toml` and running
`uv lock`); `uv.lock` is committed together with `pyproject.toml`. CI installs them with
`uv sync --locked`, the image with `uv sync --frozen --no-dev`.

Tests are hermetic (sqlite, dummy environment in `tests/conftest.py`), so no live
Postgres/Redis is needed. Pipeline tests require the pipeline dependencies to be
installed (geopandas, nltk, etc.).

Integration tests (`tests/integration`, marker `integration`) run against real
PostgreSQL, Redis and MinIO: Alembic migrations build the model schema and roll back, a task goes
through its statuses in the database, the Celery broker accepts a message, task files pass through
MinIO. Without the `PZZ_TEST_*` variables they are skipped:

```bash
PZZ_TEST_DATABASE_URL=postgresql+psycopg://pzz:pzz@localhost:5432/pzz \
PZZ_TEST_REDIS_URL=redis://localhost:6379/0 \
PZZ_TEST_MINIO_ENDPOINT=localhost:9000 PZZ_TEST_MINIO_ACCESS_KEY=minioadmin \
PZZ_TEST_MINIO_SECRET_KEY=minioadmin \
uv run pytest -m integration tests/integration
```

## CI and versions

- `tests.yml` — on every PR to `dev`/`main`: unit tests, integration tests (`integration.yml`,
  also nightly) and a black check (mismatches are fixed by a commit to the PR branch).
- `pr-autofill.yml` — PR title `vX.Y.Z (branch)` (the version it will be merged as) and a list
  of commits in the "Commits" section at the end of the description. Text written by the PR author
  is not changed: only that section is updated.
- `version-bump.yml` — the version is bumped in the PR branch before merging into `dev`: enable
  **Enable auto-merge**, and a `bump:` commit will update `pyproject.toml`, `service/__version__.py`,
  `uv.lock` and `CHANGELOG.md` (label `major` → major, branches `feat/*`/`feature/*` → minor,
  everything else → patch), after which the `version` status lets the merge through. Without
  auto-merge, `version` waits.
  - **Merging into `dev` is only possible via Enable auto-merge.** The ruleset on `dev` requires the
    `version` status, so the regular Merge button waits for it. Every PR gets a `bump:` commit from
    the bot; when `dev` moves ahead, the bot merges `dev` into the branches of PRs queued for
    auto-merge again and bumps their version.
  - **The `VERSION_STATUS_TOKEN` token.** A repository secret holding a fine-grained PAT (access to
    this repository, Commit statuses: write permission) or a classic PAT with the `repo:status`
    scope. It is used to set the final `version` status: a merge that auto-merge performs after a
    check by `GITHUB_TOKEN` does not trigger workflows, so the dev release would not start. The
    token is issued and renewed by the repository maintainer; when it expires, `version` stops
    passing, and a new token must be issued and the secret updated (Settings → Secrets and
    variables → Actions).
- `release.yml` — a merge into `main` tags the version from `dev` and creates a draft release from
  the CHANGELOG.
- The version is shown in `GET /health` and in OpenAPI.

---

## Deployment

The CI pipeline [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)
(`workflow_dispatch`) builds the image, pushes it to the registry and brings up the full stack
via [`docker-compose.actions.yml`](docker-compose.actions.yml):
`postgres + redis + migrate + api + worker + beat + mcp`.

In production, migrations are run by a separate one-shot `migrate` service
(`RUN_MIGRATIONS_ON_STARTUP=false` for api).

---

## Checking VRI against the territory's PZZ

`POST /pzz/regulations/check-vri` (MCP `check_vri_in_pzz`) tells the compliance-checking
agent whether a permitted use type (VRI) is allowed in a zone under the territory's PZZ.
The request contains one item from each group:

- territory: `project_id` (the project's territory) or `territory_id` (urban_api);
- use type: `vri_code` ("2.1"), `physical_object_type_id` (urban_api object type;
  a residential building is resolved by its number of floors, as in the scenario flow) or `service_type_id`;
- zone: `pzz_zone_code` (an exact PZZ zone, "Ж-2.15") or `functional_zone_type_id`
  (urban_api functional zone type);
- optionally `floors`, `height` — to check the maximum permitted parameters.

**Where the PZZ comes from.** The territory is provided by urban_api with the user's token: the
project geometry (`/projects/{id}/territory`) → the deepest territory that fully covers it
(`/common_territory`) → its parents. NormGraph (`NORMGRAPH_BASE_URL`, service token
`KEYCLOAK_*`) returns the PZZ of these territories (`/regulations/documents`,
`/regulations/zones`); the PZZ of the deepest one is used (the settlement, not the district).

**Functional zone type → PZZ zones.** One urban_api type corresponds to several PZZ zones.
Which ones exactly is decided by the chat language model (`LLM_BACKEND`, `CHAT_MODEL`), based on
the name, group and main VRIs of each zone in the document and the type descriptions in the
built-in template (`functional_zones_to_pzz_mapping.json`). The model answers three times in
parallel, and a zone is assigned to the type by majority vote. Subzones (Ж-1.10) follow their
parent zone (Ж-1); a subtype zone ("Low-rise residential zone") also follows the general type.
The answer is cached per document revision (a hash of its zones). Without a model, zones are
matched by code prefix (Ж — residential, О/ОИ/ОД — public and business, П/ПП/И — industrial,
Т/ТД/И — transport, …). The matching method is shown in `zone.mapping.method`
(`llm` / `code_prefix`), and the zones themselves in `zone.mapping.zone_codes`.

The **verdict** is checked for each zone (`zones[]`: verdict, reason, maximum permitted
parameters) and then aggregated:

- permitted in all zones — `allowed_main` / `allowed_conditional` /
  `allowed_auxiliary`, according to the weakest section;
- not permitted in any zone — `not_allowed`;
- permitted in some zones — `depends_on_zone` ("Зависит от зоны ПЗЗ", "Depends on the PZZ zone"):
  the decision depends on the PZZ zone where the object is located;
- no VRI for the object, no zones of this type in the document, or the zones' VRIs could not be
  read — `unclear`.

Number of floors and height are compared with the zone's maximum permitted parameters (taking into
account rows "for use type …", "except …", residential / non-residential development; "not subject
to establishment" imposes no limit): `parameters.status` — "Соответствует" (Compliant) /
"Превышены" (Exceeded) / "Не проверено" (Not checked).

No PZZ for the territory — 404; urban_api / NormGraph unavailable — 502;
`NORMGRAPH_BASE_URL` or the service account not configured — 503.

## Configuration

All settings come from environment variables / `.env.development`
(see `service/settings.py` and `.env.example`). Key ones: `DATABASE_URL`,
`REDIS_URL`, `LLM_BACKEND` + models, `FILESERVER_*` (MinIO), `URBAN_API_BASE_URL`.
For the chat endpoints: `CHAT_STORAGE_BASE_URL` (conversation history; empty — persistence disabled),
`CHAT_MODEL` (default chat model on the host of the selected `LLM_BACKEND`), `CHAT_SYSTEM_PROMPT_PATH`,
`KEYCLOAK_*` (service token for writing history, see [Authentication](#authentication)).
Checking VRI against the territory's PZZ: `NORMGRAPH_BASE_URL` (empty — the check is unavailable),
`NORMGRAPH_TIMEOUT_SECONDS`; NormGraph is called with the same service token.
Secrets are not committed to the repository.

### vLLM request priorities

The service sends `priority` in the JSON of every `/v1/chat/completions` request,
following the shared priority matrix:

| Model call | Class | Variable | Default |
| --- | --- | --- | --- |
| Streamed chat answer / PZZ report | live / blocking | `VLLM_CHAT_PRIORITY` | `-100` |
| VRI classification, PZZ check, candidate re-scoring, attribute parsing, zone and name mapping | live-long / blocking | `VLLM_REQUEST_PRIORITY` | `-60` |

Lower numbers are handled first: the answer to the user (`-100`) goes before the
checks (`-60`), and both go before regular requests with `0`. The `priority` parameter
of an API task (1–10) controls worker capacity reservation and does not change
these vLLM classes. The text report file is produced without the model.

**Before updating the service**, add the `--scheduling-policy priority` flag to the
launch command of the external `vllm serve <model> ...` server and restart it.
vLLM is not started here: changing the application's Docker Compose will not switch
its scheduler. Then set in `.env.development` (for Actions, in `ENV_APP`):

```dotenv
VLLM_REQUEST_PRIORITY=-60
VLLM_CHAT_PRIORITY=-100
```

Restart the API and the LLM workers. For a server still on the old `fcfs` scheduler,
temporarily set **both** variables to `0`: vLLM rejects a non-zero priority without
priority scheduling. The application itself does not lower the priority automatically
when the server returns an error. Priority changes the order of service, but does not
fix model or network errors or a GPU shortage.

Contract: [request priority](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/),
[vLLM scheduler](https://docs.vllm.ai/en/latest/cli/serve/#scheduling-policy).

## Text reports

After an operation completes, the chat additionally emits a `file` event with
`name: "report"`, `role: "report"`, `mime_type: "text/plain"`, `filename: "Отчёт.txt"`
("Report.txt"). The regular text answer stays in the stream. The link to the report is
stored next to it in the ChatStorage history as a `kind: "file"` part; the file contents
are not copied into the history.

The descriptor carries two durable links relative to the `source_service` base URL:

- `url`: `GET /reports/id` — UTF-8 text for viewing (`Content-Disposition: inline`).
- `download_url`: the same address with `?download=true` — download (`attachment; filename="report.txt"`).

The descriptor is located in the `content` of the `file` event.
The report is built from the stored task result: PZZ check by land plots/buildings/scenario,
classification or site selection. It includes the full summary, all object rows and the
standards, without the text truncation applied for the LLM. `id` is the task's `external_id`.
Until the task has finished, the API returns 409. The file is available regardless of the LLM
and ChatStorage; `include_report=false` hides the structured SSE event but not the file attachment.

The frontend should read the report as text, not as GeoJSON or HTML. To display it,
`fetch(url).text()` and a text component that preserves line breaks are enough.
Links go through the service API, so the browser never needs access to the internal MinIO.
The access model is the same as for the existing `/files/...`: the task UUID is the access key.
Deleting the task/result makes the report unavailable.

The report is written in Russian, following the layout of a regulatory compliance check:
headings, a summary metrics table and separate detail sections. In PZZ reports, objects
are grouped by compliance and by whether they need manual review; a classification result
is never presented as a PZZ compliance check.
The text is saved as UTF-8 TXT with Markdown markup suitable for rendering on the
frontend. Internal API fields are omitted; parameter names, zone names and system
statuses are translated. Original zone codes, VRI codes, document codes and metric values
are kept as is. These changes affect only how the report is presented; the JSON and SSE
schemas are unchanged.

For download, the Russian file name `Отчёт.txt` is sent via the `filename*` header
parameter; `filename` carries an ASCII-compatible fallback name.
