from functools import lru_cache
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from iduconfig import Config


def _get_required_env(config: Config, key: str) -> str:
    value = config.get(key)
    if not value:
        raise ValueError(f"{key} is required")
    return value


class Settings(BaseSettings):
    """Application settings loaded from environment and .env file."""

    app_name: str = "PZZ Pipeline Service"
    database_url: str = Field(...)
    redis_url: str = Field(...)
    pipeline_module: str = Field(default="pipeline_modules.pipeline_v25")
    pipeline_callable: str = Field(
        default="pipeline_modules.pipeline_impl:run_pipeline"
    )
    pipeline_runner_mode: str = Field(default="subprocess")
    pipeline_runner_fallback_enabled: bool = Field(default=True)
    pipeline_runner_fallback_mode: str = Field(default="subprocess")
    outputs_dir: str = Field(default="outputs")
    outputs_cleanup_max_age_hours: int = Field(default=168)
    outputs_cleanup_interval_seconds: int = Field(default=3600)
    reconcile_interval_seconds: int = Field(default=60)
    task_soft_time_limit_seconds: int = Field(default=6600)
    task_time_limit_seconds: int = Field(default=7200)
    max_upload_bytes: int = Field(default=200 * 1024 * 1024)
    task_inputs_dir: str = Field(default="task_inputs")
    default_pzz_zone_labels_path: str = Field(
        default="data/pzz_zone_llm_labels_template.json"
    )
    default_vri_classifier_path: str = Field(
        default="data/rosreestr_vri_classifier_2024_12_24.json"
    )
    default_services_hierarchy_path: str = Field(
        default="data/services_hierarchy.json"
    )
    default_physical_objects_hierarchy_path: str = Field(
        default="data/physical_objects_hierarchy.json"
    )
    default_fz_to_pzz_mapping_path: str = Field(
        default="data/functional_zones_to_pzz_mapping.json"
    )
    # Scenario tasks (urban_api-backed, controlled vocabularies) classify
    # deterministically via dictionary lookups instead of the LLM pipeline.
    # Set SCENARIO_DETERMINISTIC=false to fall back to the LLM pipeline.
    scenario_deterministic: bool = Field(default=True)
    physical_object_type_to_vri_path: str = Field(
        default="data/physical_object_type_to_vri.json"
    )
    priority_max_sum_default: int = 20

    run_migrations_on_startup: bool = Field(default=True)

    # Port on which the Celery worker exposes its Prometheus metrics. Task
    # metrics (queue_wait, run duration, failures, retries) are recorded in
    # the worker process, so they cannot be served by the API's /metrics —
    # the worker starts its own HTTP exposition server on this port, scraped
    # by Prometheus as a separate target.
    worker_metrics_port: int = Field(default=9100)

    # Bearer-token verification (Keycloak JWT via JWKS). Opt-in: when
    # auth_verify is false (default) tokens are accepted without signature
    # checks (dev/test, or when an upstream gateway already validated them).
    # Set AUTH_VERIFY=true + AUTH_SERVER_URL=<realm url> in prod.
    auth_verify: bool = Field(default=False)
    auth_server_url: str = Field(default="")  # https://.../realms/<realm>
    auth_client_id: str = Field(default="")
    auth_verify_aud: bool = Field(default=True)
    auth_valid_audiences: str = Field(default="")  # comma-separated
    auth_user_cache_ttl: int = Field(default=300)
    auth_user_cache_size: int = Field(default=10_000)
    auth_jwks_cache_ttl: int = Field(default=600)
    auth_timeout_seconds: int = Field(default=5)

    urban_api_base_url: str = Field(default="")
    urban_api_timeout_seconds: float = Field(default=30.0)

    fileserver_endpoint: str = Field(default="")
    fileserver_access_key: str = Field(default="")
    fileserver_secret_key: str = Field(default="")
    fileserver_bucket_name: str = Field(default="")
    fileserver_secure: bool = Field(default=False)

    llm_backend: str = Field(...)
    ollama_base_url: str = Field(...)
    vllm_base_url: str = Field(...)
    vllm_api_key: str = Field(...)
    embed_model: str = Field(...)
    generate_model: str = Field(...)
    top_k: int = Field(default=10)
    embed_batch_size: int = Field(default=32)

    model_config = SettingsConfigDict(
        env_file=".env.development",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    if not os.getenv("APP_ENV"):
        os.environ["APP_ENV"] = "development"

    config = Config()
    return Settings(
        database_url=_get_required_env(config, "DATABASE_URL"),
        redis_url=_get_required_env(config, "REDIS_URL"),
        llm_backend=_get_required_env(config, "LLM_BACKEND"),
        ollama_base_url=_get_required_env(config, "OLLAMA_BASE_URL"),
        vllm_base_url=_get_required_env(config, "VLLM_BASE_URL"),
        vllm_api_key=_get_required_env(config, "VLLM_API_KEY"),
        embed_model=_get_required_env(config, "EMBED_MODEL"),
        generate_model=_get_required_env(config, "GENERATE_MODEL"),
        urban_api_base_url=(config.get("URBAN_API_BASE_URL") or "").rstrip("/"),
        urban_api_timeout_seconds=float(config.get("URBAN_API_TIMEOUT_SECONDS") or "30"),
        fileserver_endpoint=config.get("FILESERVER_ENDPOINT") or "",
        fileserver_access_key=config.get("FILESERVER_ACCESS_KEY") or "",
        fileserver_secret_key=config.get("FILESERVER_SECRET_KEY") or "",
        fileserver_bucket_name=config.get("FILESERVER_BUCKET_NAME") or "",
        fileserver_secure=(config.get("FILESERVER_SECURE") or "").lower() in {"1", "true", "yes", "on"},
    )