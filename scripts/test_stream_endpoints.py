"""Exercise all three SSE streaming endpoints and dump what each returns.

Scenario (deterministic) + pzz-check + classify-only (LLM). For each flow it
records the event sequence and a sample of every event type, writing a readable
report to stdout and a JSON to scripts/_stream_results.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

API = "http://localhost:8300"
ROOT = Path(__file__).resolve().parents[1]
CAD = r"G:/Projects/LLM_TASK/data/Долинск/merged_dolinsk_layers.geojson"
PZZ = r"G:/Projects/LLM_TASK/data/ПЗЗ/pzz_layers.geojson"


def token() -> str:
    for ln in (ROOT / ".env.development").read_text(encoding="utf-8").splitlines():
        if ln.startswith("URBAN_API_TOKEN="):
            return ln.split("=", 1)[1].strip().strip('"')
    return ""


def consume(label: str, *, url: str, data: dict, files=None, headers=None) -> dict:
    t0 = time.time()
    out: dict = {"label": label, "sequence": [], "events": {}, "statuses": []}
    with httpx.Client(timeout=httpx.Timeout(360.0, read=360.0)) as c:
        with c.stream("POST", url, data=data, files=files, headers=headers or {}) as r:
            out["http_status"] = r.status_code
            out["content_type"] = r.headers.get("content-type")
            if r.status_code != 200:
                out["error_body"] = r.read().decode("utf-8", "replace")[:400]
                return out
            ev = None
            for line in r.iter_lines():
                if line.startswith("event:"):
                    ev = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    payload = line.split(":", 1)[1].strip()
                    out["sequence"].append(ev)
                    dt = round(time.time() - t0, 1)
                    try:
                        d = json.loads(payload)
                    except Exception:
                        d = payload
                    if ev == "status" and isinstance(d, dict):
                        out["statuses"].append((dt, d.get("status")))
                    if ev == "task":
                        out["events"]["task"] = {"at": dt, "external_id": d.get("external_id"), "status": d.get("status")}
                    elif ev == "task_event":
                        out["events"].setdefault("task_event_first", {"at": dt, "data": d})
                        out["events"]["task_event_count"] = out["events"].get("task_event_count", 0) + 1
                    elif ev == "geojson" and isinstance(d, dict):
                        feats = d.get("features") or []
                        sample = feats[0].get("properties", {}) if feats else {}
                        out["events"]["geojson"] = {
                            "at": dt, "type": d.get("type"), "features": len(feats),
                            "geom_type": (feats[0].get("geometry") or {}).get("type") if feats else None,
                            "sample_property_keys": list(sample.keys()),
                            "sample_verdict": sample.get("Вердикт_ПЗЗ"),
                            "sample_matched_vri": sample.get("Код_подобранного_ВРИ"),
                            "sample_top1": sample.get("PZZ_NOT_ALLOWED_TOP1_CANDIDATE"),
                        }
                    elif ev == "report" and isinstance(d, dict):
                        zones = d.get("zones") or []
                        out["events"]["report"] = {
                            "at": dt, "summary": d.get("summary"),
                            "zones": len(zones),
                            "chat_message_head": (d.get("chat_message") or "")[:160],
                        }
                    elif ev == "error":
                        out["events"]["error"] = d
                    if ev == "done":
                        out["events"]["done"] = {"at": dt, "data": d}
                        break
    return out


def main() -> None:
    results = []

    # 1) scenario (deterministic)
    results.append(consume(
        "scenario 843 classify/stream",
        url=f"{API}/scenarios/843/classify/stream",
        data={"year": "2025", "source": "User", "force_recompute": "true", "group_by": "zone"},
        headers={"Authorization": f"Bearer {token()}"},
    ))

    # 2) upload pzz-check (LLM)
    results.append(consume(
        "pzz-check/stream (Dolinsk)",
        url=f"{API}/tasks/pzz-check/stream",
        files={
            "cadastral_feature_collection_file": ("cad.geojson", open(CAD, "rb"), "application/geo+json"),
            "pzz_zones_feature_collection_file": ("pzz.geojson", open(PZZ, "rb"), "application/geo+json"),
        },
        data={"cadastral_vri_col": "Вид_разрешенного_исп", "pzz_zone_code_col": "Индекс_зоны",
              "pzz_zone_name_col": "Код_объекта", "force_recompute": "true", "group_by": "zone"},
    ))

    # 3) upload classify-only (LLM, no zones)
    results.append(consume(
        "classify-only/stream (Dolinsk)",
        url=f"{API}/tasks/classify-only/stream",
        files={"cadastral_feature_collection_file": ("cad.geojson", open(CAD, "rb"), "application/geo+json")},
        data={"cadastral_vri_col": "Вид_разрешенного_исп", "force_recompute": "true"},
    ))

    (ROOT / "scripts" / "_stream_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for r in results:
        print("=" * 70)
        print(r["label"], "| HTTP", r.get("http_status"), r.get("content_type"))
        print("  sequence:", " -> ".join(r["sequence"]))
        print("  statuses:", r["statuses"])
        for k in ("task", "geojson", "report", "error", "done"):
            if k in r["events"]:
                print(f"  {k}:", json.dumps(r["events"][k], ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
