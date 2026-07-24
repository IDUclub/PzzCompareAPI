"""Drive POST /tasks/chat/stream and print the SSE event sequence."""

import json
import time

import httpx

BASE = "http://localhost:8300"
CAD = r"G:\Projects\LLM_TASK\data\Сокол\merged_sokol_layers.geojson"
ZONES = r"G:\Projects\LLM_TASK\data\ПЗЗ\pzz_layers.geojson"
TOKEN = open("scripts/_test_token.txt", encoding="utf-8").read().strip()

data = {
    "user_query": "Какие объекты стоят не в своей функциональной зоне? Ответь кратко.",
    "cadastral_vri_col": "Вид_разрешенного_исп",
    "pzz_zone_code_col": "Индекс_зоны",
    "pzz_zone_name_col": "Код_объекта",
    "group_by": "zone",
    "Idempotency-Key": "sokol-chat-test-1",
}

t0 = time.time()
chunks: list[str] = []
chat_id = None


def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


with open(CAD, "rb") as cf, open(ZONES, "rb") as zf:
    files = {
        "cadastral_feature_collection_file": (
            "cad.geojson",
            cf,
            "application/geo+json",
        ),
        "pzz_zones_feature_collection_file": (
            "zones.geojson",
            zf,
            "application/geo+json",
        ),
    }
    with httpx.Client(timeout=httpx.Timeout(None)) as client:
        with client.stream(
            "POST",
            f"{BASE}/tasks/chat/stream",
            files=files,
            data=data,
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as resp:
            log("HTTP", resp.status_code, resp.headers.get("content-type"))
            if resp.status_code != 200:
                log("BODY", resp.read().decode("utf-8", "replace")[:500])
                raise SystemExit(1)
            event = None
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    event = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    raw = line[len("data:") :].strip()
                    try:
                        d = json.loads(raw)
                    except json.JSONDecodeError:
                        d = raw
                    if event == "chunk" and isinstance(d, dict):
                        chunks.append(d.get("content", {}).get("text", ""))
                        if d.get("content", {}).get("done"):
                            log("chunk DONE")
                    elif event == "file":
                        c = d.get("content", {})
                        log(
                            "file",
                            c.get("role"),
                            c.get("name"),
                            "url=",
                            c.get("url"),
                            "download=",
                            (c.get("download_url") or "")[:60],
                        )
                    elif event == "service_event":
                        ev = d.get("content", {}).get("event", {})
                        chat_id = ev.get("chat_id")
                        log("service_event chat_created chat_id=", chat_id)
                    elif event == "object_zone_fit":
                        s = d.get("summary") if isinstance(d, dict) else None
                        log("object_zone_fit summary=", s)
                    elif event == "status":
                        log("status ->", d.get("status") if isinstance(d, dict) else d)
                    elif event == "task":
                        log(
                            "task external_id=",
                            d.get("external_id") if isinstance(d, dict) else d,
                        )
                    elif event == "done":
                        log("done", d)
                    elif event == "error":
                        log("ERROR", d)
                    elif event == "task_event":
                        pass  # too chatty
                    else:
                        log(event, str(d)[:120])

answer = "".join(chunks)
with open("scripts/_chat_answer.txt", "w", encoding="utf-8") as fh:
    fh.write(answer)
log("=== ASSISTANT ANSWER written to scripts/_chat_answer.txt, len=", len(answer))
log("chat_id:", chat_id)
