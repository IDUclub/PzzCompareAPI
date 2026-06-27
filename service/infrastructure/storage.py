"""Object storage abstraction with MinIO backend and local fallback.

Stored paths use a ``minio://`` prefix when the object lives in MinIO; plain
filesystem paths have no scheme. This lets the rest of the codebase pass
the stored path around opaquely — only this module knows how to materialise
it locally or upload to remote storage.

The backend is chosen at startup based on settings: when both
``FILESERVER_ACCESS_KEY`` and ``FILESERVER_SECRET_KEY`` are non-empty MinIO
is used; otherwise a no-op local backend keeps the existing behaviour.
"""
from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path


_MINIO_SCHEME = "minio://"


def is_remote_path(stored_path: str | None) -> bool:
    """Return True when the stored path refers to an object in MinIO."""
    return bool(stored_path) and stored_path.startswith(_MINIO_SCHEME)


class ObjectStorage(ABC):
    """Backend for task input/output blobs."""

    @abstractmethod
    def is_remote(self) -> bool:
        """Whether stored paths are remote (MinIO) or local."""

    @abstractmethod
    def upload_file(self, local_path: str, object_key: str) -> str:
        """Upload local file. Returns the canonical stored path."""

    @abstractmethod
    def download_file(self, stored_path: str, local_path: str) -> str:
        """Materialise stored object at local_path. Returns local_path."""

    @abstractmethod
    def delete(self, stored_path: str) -> None:
        """Best-effort delete; never raises."""

    def presigned_url(self, stored_path: str, expires_seconds: int = 3600) -> str | None:
        """Return a time-limited direct download URL, or None if unsupported.

        Only the remote (MinIO) backend can mint one; local storage returns
        None so callers fall back to a backend-served download.
        """
        return None


class LocalStorage(ObjectStorage):
    """Filesystem-backed storage that preserves the legacy behaviour."""

    def is_remote(self) -> bool:
        return False

    def upload_file(self, local_path: str, object_key: str) -> str:
        return str(Path(local_path).resolve())

    def download_file(self, stored_path: str, local_path: str) -> str:
        src = Path(stored_path).resolve()
        dst = Path(local_path).resolve()
        if src == dst:
            return str(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return str(dst)

    def delete(self, stored_path: str) -> None:
        try:
            Path(stored_path).unlink(missing_ok=True)
        except OSError:
            pass


class MinioStorage(ObjectStorage):
    """MinIO / S3-compatible object storage."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ) -> None:
        from minio import Minio

        if not endpoint or not bucket:
            raise ValueError("MinIO endpoint and bucket are required")
        self._bucket = bucket
        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)

    def is_remote(self) -> bool:
        return True

    @staticmethod
    def _strip_scheme(path: str) -> str:
        return path[len(_MINIO_SCHEME):] if path.startswith(_MINIO_SCHEME) else path

    def upload_file(self, local_path: str, object_key: str) -> str:
        self._client.fput_object(self._bucket, object_key, local_path)
        return f"{_MINIO_SCHEME}{object_key}"

    def download_file(self, stored_path: str, local_path: str) -> str:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        object_key = self._strip_scheme(stored_path)
        self._client.fget_object(self._bucket, object_key, local_path)
        return local_path

    def delete(self, stored_path: str) -> None:
        object_key = self._strip_scheme(stored_path)
        try:
            self._client.remove_object(self._bucket, object_key)
        except Exception:  # noqa: BLE001 — delete is best-effort
            pass

    def presigned_url(self, stored_path: str, expires_seconds: int = 3600) -> str | None:
        from datetime import timedelta

        object_key = self._strip_scheme(stored_path)
        try:
            return self._client.presigned_get_object(
                self._bucket, object_key, expires=timedelta(seconds=expires_seconds)
            )
        except Exception:  # noqa: BLE001 — presign failure shouldn't break the stream
            return None


@lru_cache(maxsize=1)
def get_object_storage() -> ObjectStorage:
    """Return the configured storage backend (cached for the process lifetime).

    MinIO is selected only when access_key, secret_key, endpoint and bucket
    are all provided. Otherwise the service uses local filesystem storage,
    which keeps existing dev workflows working without any configuration.
    """
    from service.settings import get_settings

    s = get_settings()
    if (
        s.fileserver_access_key
        and s.fileserver_secret_key
        and s.fileserver_endpoint
        and s.fileserver_bucket_name
    ):
        return MinioStorage(
            endpoint=s.fileserver_endpoint,
            access_key=s.fileserver_access_key,
            secret_key=s.fileserver_secret_key,
            bucket=s.fileserver_bucket_name,
            secure=s.fileserver_secure,
        )
    return LocalStorage()
