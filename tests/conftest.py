"""Test bootstrap.

Importing ``service`` builds a SQLAlchemy engine at module import time
(``service/db.py``) and ``get_settings()`` requires a handful of env vars.
We set safe, hermetic defaults here — before any test module imports
``service`` — so the suite runs without a live Postgres/Redis or a real
``.env``. ``load_dotenv`` (used by iduconfig) does not override variables
already present in the environment, so these win over any ``.env.*`` file.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["APP_ENV"] = os.environ.get("APP_ENV", "development")
os.environ["DATABASE_URL"] = "sqlite:///./test_pzz_pipeline.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"
os.environ.setdefault("LLM_BACKEND", "ollama")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("VLLM_BASE_URL", "http://localhost:8001")
os.environ.setdefault("VLLM_API_KEY", "test")
os.environ.setdefault("EMBED_MODEL", "test-embed")
os.environ.setdefault("GENERATE_MODEL", "test-generate")
os.environ["RUN_MIGRATIONS_ON_STARTUP"] = "false"
