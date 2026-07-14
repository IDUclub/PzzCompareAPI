"""Stress the scenario classify endpoint while probing /health responsiveness.

Fires N concurrent classify requests (unique idempotency keys -> distinct
tasks) and, on a separate thread, hits /health every second. Records both so
we can see whether the event loop stays responsive under a burst.
"""
import json
import threading
import time
from pathlib import Path

import httpx

API = "http://localhost:8300"
ROOT = Path(__file__).resolve().parents[1]
N = 10
tok = [l.split("=", 1)[1].strip().strip('"') for l in (ROOT / ".env.development").read_text(encoding="utf-8").splitlines() if l.startswith("URBAN_API_TOKEN=")][0]

t0 = time.time()
health: list = []
results: list = []
stop = threading.Event()


def health_poll():
    while not stop.is_set():
        t = time.time()
        try:
            r = httpx.get(f"{API}/health", timeout=5.0)
            health.append([round(time.time() - t0, 1), r.status_code, round(time.time() - t, 2)])
        except Exception as e:
            health.append([round(time.time() - t0, 1), "ERR", round(time.time() - t, 2), type(e).__name__])
        time.sleep(1.0)


def fire(i):
    h = {"Authorization": f"Bearer {tok}", "Idempotency-Key": f"fix-{int(t0)}-{i}"}
    t = time.time()
    try:
        r = httpx.post(f"{API}/scenarios/843/classify",
                       data={"year": "2025", "source": "User", "force_recompute": "true", "priority": "1"},
                       headers=h, timeout=180.0)
        results.append([i, r.status_code, round(time.time() - t, 1),
                        (r.json().get("status") if r.status_code == 200 else r.text[:40])])
    except Exception as e:
        results.append([i, "ERR", round(time.time() - t, 1), type(e).__name__])


hp = threading.Thread(target=health_poll)
hp.start()
fts = [threading.Thread(target=fire, args=(i,)) for i in range(N)]
[f.start() for f in fts]
[f.join() for f in fts]
stop.set()
hp.join()

ok = sum(1 for r in results if r[1] == 200)
hlat = [h[2] for h in health]
hbad = [h for h in health if h[1] != 200]
out = {
    "classify_ok": f"{ok}/{N}",
    "classify_latency_s": sorted(r[2] for r in results),
    "results": sorted(results),
    "health_probes": len(health),
    "health_non200": len(hbad),
    "health_latency_max_s": max(hlat) if hlat else None,
    "health_timeline": health,
}
(ROOT / "scripts" / "_stress_fix.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: out[k] for k in ("classify_ok", "classify_latency_s", "health_probes", "health_non200", "health_latency_max_s")}, ensure_ascii=False))
