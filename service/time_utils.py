from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return a timezone-aware UTC ``datetime``.

    Using a helper avoids the deprecated ``datetime.utcnow()`` and keeps a
    single source of truth so timestamps written to the database are always
    timezone-aware.
    """
    return datetime.now(timezone.utc)
