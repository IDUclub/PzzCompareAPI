"""Smoke-test the deterministic scenario path against the local stack.

POST /scenarios/{id}/classify -> poll status -> GET object-zone-fit, for each
of scenarios 843/844/845 (2025/User). Reads the urban_api token from
.env.development. Run after `docker compose -f docker-compose.localtest.yml up`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

API = "http://localhost:8300"
ROOT = Path(__file__).resolve().parents[1]


def token() -> str:
    for ln in (ROOT / ".env.development").read_text(encoding="utf-8").splitlines():
        if ln.startswith("URBAN_API_TOKEN="):
            return ln.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no URBAN_API_TOKEN in .env.development")


def run_one(client: httpx.Client, hdr: dict, sid: int) -> None:
    print(f"\n===== scenario {sid} (2025/User) =====")
    r = client.post(
        f"{API}/scenarios/{sid}/classify",
        headers=hdr,
        data={"year": 2025, "source": "User", "force_recompute": "true"},
    )
    if r.status_code != 200:
        print(f"  classify FAILED {r.status_code}: {r.text[:300]}")
        return
    ext = r.json()["external_id"]
    print(f"  classify ok -> external_id={ext}, status={r.json()['status']}")

    status, waited = "queued", 0
    while status not in ("finished", "failed") and waited < 120:
        time.sleep(2)
        waited += 2
        t = client.get(f"{API}/scenarios/{sid}/tasks/{ext}", headers=hdr).json()
        status = t.get("status", status)
    print(f"  final status={status} after ~{waited}s")
    if status == "failed":
        print(f"  error_text: {t.get('error_text')}")
        return

    rep = client.get(
        f"{API}/scenarios/{sid}/tasks/{ext}/object-zone-fit",
        params={"group_by": "zone"},
        headers=hdr,
    )
    if rep.status_code != 200:
        print(f"  report FAILED {rep.status_code}: {rep.text[:300]}")
        return
    data = rep.json()
    s = data["summary"]
    print(
        f"  SUMMARY total={s['total']} correct={s['in_correct_zone']} "
        f"wrong={s['in_wrong_zone']} unclear={s['unclear']}"
    )
    for z in data.get("zones", [])[:6]:
        zs = z["summary"]
        print(
            f"    zone «{z.get('zone_name') or z.get('zone_type_id')}»: "
            f"total={zs['total']} correct={zs['in_correct_zone']} wrong={zs['in_wrong_zone']} unclear={zs['unclear']}"
        )


def main() -> None:
    hdr = {"Authorization": f"Bearer {token()}"}
    sids = [int(x) for x in sys.argv[1:]] or [843, 844, 845]
    with httpx.Client(timeout=60) as client:
        try:
            h = client.get(f"{API}/health", timeout=5)
            print(f"health: {h.status_code} {h.text}")
        except Exception as e:
            raise SystemExit(f"API not reachable at {API}: {e}")
        for sid in sids:
            run_one(client, hdr, sid)


if __name__ == "__main__":
    main()
