import os
import sys
import time
from urllib.parse import urlparse

import psycopg


def normalize_database_url(database_url: str) -> str:
    """Convert SQLAlchemy-style psycopg URL into a psycopg-compatible URL."""
    if database_url.startswith("postgresql+psycopg://"):
        return "postgresql://" + database_url[len("postgresql+psycopg://"):]
    return database_url


def mask_database_url(database_url: str) -> str:
    """Hide password in logs."""
    parsed = urlparse(database_url)
    if not parsed.password:
        return database_url

    netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
    return parsed._replace(netloc=netloc).geturl()


def wait_for_db() -> int:
    """Poll PostgreSQL until connection becomes available or timeout expires."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL is not set", flush=True)
        return 1

    timeout_seconds = int(os.getenv("DB_WAIT_TIMEOUT", "60"))
    retry_interval_seconds = float(os.getenv("DB_WAIT_INTERVAL", "2"))

    conninfo = normalize_database_url(database_url)
    masked_conninfo = mask_database_url(conninfo)

    deadline = time.monotonic() + timeout_seconds
    attempt = 0

    print(f"Waiting for PostgreSQL: {masked_conninfo}", flush=True)

    while time.monotonic() < deadline:
        attempt += 1
        try:
            with psycopg.connect(conninfo, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    cur.fetchone()

            print(f"PostgreSQL is ready on attempt {attempt}", flush=True)
            return 0

        except Exception as exc:
            print(
                f"PostgreSQL is not ready yet on attempt {attempt}: {exc}",
                flush=True,
            )
            time.sleep(retry_interval_seconds)

    print(
        f"Timed out waiting for PostgreSQL after {timeout_seconds} seconds",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    sys.exit(wait_for_db())
