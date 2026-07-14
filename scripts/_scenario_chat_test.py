"""Drive POST /scenarios/{id}/chat/stream and print the SSE event sequence."""
import json
import time

import httpx

import os
BASE = os.environ.get("PZZ_BASE", "http://localhost:8300")
SCENARIO_ID = 604
TOKEN = open("scripts/_test_token.txt", encoding="utf-8").read().strip()

data = {
    "user_query": "Какие жилые объекты стоят не в своей функциональной зоне? Ответь кратко.",
    "year": 2023,
    "source": "PZZ",
    "group_by": "zone",
}

t0 = time.time()
chunks: list[str] = []
chat_id = None


def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


with httpx.Client(timeout=httpx.Timeout(None)) as client:
    with client.stream(
        "POST",
        f"{BASE}/scenarios/{SCENARIO_ID}/chat/stream",
        data=data,
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as resp:
        log("HTTP", resp.status_code, resp.headers.get("content-type"))
        if resp.status_code != 200:
            log("BODY", resp.read().decode("utf-8", "replace")[:600])
            raise SystemExit(1)
        event = None
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                raw = line[len("data:"):].strip()
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
                    log("file", c.get("role"), c.get("name"), "url=", c.get("url"),
                        "download=", (c.get("download_url") or "")[:55])
                elif event == "service_event":
                    ev = d.get("content", {}).get("event", {})
                    chat_id = ev.get("chat_id")
                    log("service_event chat_created chat_id=", chat_id)
                elif event == "object_zone_fit":
                    log("object_zone_fit summary=", d.get("summary") if isinstance(d, dict) else d)
                elif event == "status":
                    log("status ->", d.get("status") if isinstance(d, dict) else d)
                elif event == "task":
                    log("task external_id=", d.get("external_id") if isinstance(d, dict) else d)
                elif event == "done":
                    log("done", d)
                elif event == "error":
                    log("ERROR", d)
                elif event == "task_event":
                    pass
                else:
                    log(event, str(d)[:120])

answer = "".join(chunks)
with open("scripts/_scenario_answer.txt", "w", encoding="utf-8") as fh:
    fh.write(answer)
log("=== ANSWER written to scripts/_scenario_answer.txt, len=", len(answer))
log("chat_id:", chat_id)
