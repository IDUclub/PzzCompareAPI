from __future__ import annotations

import threading

import numpy as np
import requests
from iduconfig import Config

from .common import *
from .text_utils import normalize_text, safe_json_loads

config = Config()

class VLLMEmptyContentError(ValueError):
    """Raised when vLLM returned no final assistant content."""


class VLLMTruncatedReasoningError(VLLMEmptyContentError):
    """Raised when vLLM spent all tokens on reasoning before final output."""


def parse_think_value(value: Any) -> Any:
    """Parse think configuration preserving reasoning levels."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    text_lc = text.lower()
    if text_lc in {"1", "true", "yes", "on"}:
        return True
    if text_lc in {"0", "false", "no", "off"}:
        return False
    if text_lc in {"low", "medium", "high", "auto"}:
        return text_lc

    return text

class VectorizerClient:
    """Client for OpenAI-compatible embeddings endpoint with batch support."""

    def __init__(self, url: str, model: str, client_cert_path: Optional[str] = None, client_key_path: Optional[str] = None, ca_cert_path: Optional[str] = None, timeout: int = 300) -> None:
        self.url = url
        self.model = model
        self.client_cert_path = client_cert_path
        self.client_key_path = client_key_path
        self.ca_cert_path = ca_cert_path
        self.timeout = timeout
        self.max_parallel_requests = max(1, int(config.get("EMBED_MAX_PARALLEL_REQUESTS") or 1))

    def _build_request_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"timeout": self.timeout}
        if self.client_cert_path and self.client_key_path and os.path.exists(self.client_cert_path) and os.path.exists(self.client_key_path):
            kwargs["cert"] = (self.client_cert_path, self.client_key_path)
        if self.ca_cert_path and os.path.exists(self.ca_cert_path):
            kwargs["verify"] = self.ca_cert_path
        return kwargs

    def _embed_batch(self, batch_index: int, batch: list[str]) -> tuple[int, list[list[float]]]:
        payload = {"input": batch, "model": self.model, "encoding_format": "float"}
        response = requests.post(self.url, json=payload, **self._build_request_kwargs())
        response.raise_for_status()
        data = response.json()
        items = data.get("data") or []
        if len(items) != len(batch):
            raise ValueError(f"Unexpected embedding count: expected={len(batch)}, got={len(items)}")
        ordered = sorted(items, key=lambda item: int(item.get("index", 0)))
        vectors = [item["embedding"] for item in ordered]
        return (batch_index, vectors)

    def embed_many(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Compute normalized embeddings for a list of texts in batches."""
        cleaned = [normalize_text(text) for text in texts]
        if not cleaned:
            return np.zeros((0, 0), dtype=np.float32)
        batches: list[list[str]] = [
            cleaned[start:start + batch_size]
            for start in range(0, len(cleaned), batch_size)
        ]
        all_vectors: list[list[float]] = []
        if self.max_parallel_requests == 1 or len(batches) == 1:
            for batch_index, batch in enumerate(batches):
                _, vectors = self._embed_batch(batch_index, batch)
                all_vectors.extend(vectors)
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_parallel_requests, len(batches))) as executor:
                futures = [
                    executor.submit(self._embed_batch, batch_index, batch)
                    for batch_index, batch in enumerate(batches)
                ]
                completed: list[tuple[int, list[list[float]]]] = [
                    future.result() for future in as_completed(futures)
                ]
            for _, vectors in sorted(completed, key=lambda item: item[0]):
                all_vectors.extend(vectors)
        matrix = np.asarray(all_vectors, dtype=np.float32)
        if matrix.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class OllamaLLMClient:
    """Unified Ollama client with auto runtime selection for different model families."""

    def __init__(self, base_url: str, mode: str = "auto", timeout: int = 900, default_model: Optional[str] = None, keep_alive: str = "15m", temperature: float = 0.0, num_ctx: int = 16384, num_predict: int = 16384, think: Any = "auto", runtime_presets: Optional[dict[str, dict[str, Any]]] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/api/chat"
        self.generate_url = f"{self.base_url}/api/generate"
        self.mode = mode
        self.timeout = timeout
        self.default_model = default_model
        self.keep_alive = keep_alive
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.think = think
        self.runtime_presets = runtime_presets or {}

    def _resolve_runtime(self, model: str) -> tuple[str, Any]:
        model_lc = normalize_text(model).lower()
        preset_mode = None
        preset_think = None
        for prefix, preset in self.runtime_presets.items():
            prefix_lc = normalize_text(prefix).lower()
            if prefix_lc and model_lc.startswith(prefix_lc):
                preset_mode = preset.get("mode")
                preset_think = preset.get("think")
                break
        resolved_mode = self.mode
        if normalize_text(str(resolved_mode)).lower() == "auto":
            resolved_mode = preset_mode or "chat"
        requested_think = self.think
        if isinstance(requested_think, str) and requested_think.strip().lower() == "auto":
            requested_think = preset_think if preset_think is not None else False
        thinking_capable_prefixes = ("qwen3", "gpt-oss", "deepseek-r1", "deepseek-v3.1")
        supports_thinking = any((model_lc.startswith(prefix) for prefix in thinking_capable_prefixes))
        if not supports_thinking:
            requested_think = None
        elif isinstance(requested_think, str):
            think_lc = requested_think.strip().lower()
            if not think_lc:
                requested_think = None
            elif model_lc.startswith("gpt-oss"):
                if think_lc not in {"low", "medium", "high"}:
                    requested_think = True if think_lc in {"true", "1", "yes", "on"} else False
            elif think_lc in {"low", "medium", "high"}:
                requested_think = True
            elif think_lc in {"true", "1", "yes", "on"}:
                requested_think = True
            elif think_lc in {"false", "0", "no", "off"}:
                requested_think = False
            else:
                requested_think = None
        return (normalize_text(str(resolved_mode)).lower(), requested_think)

    def _base_payload_options(self) -> dict[str, Any]:
        return {"keep_alive": self.keep_alive, "options": {"temperature": self.temperature, "num_ctx": self.num_ctx, "num_predict": self.num_predict}}

    def _chat_json(self, model: str, system_prompt: str, user_prompt: str, schema: dict[str, Any], think: Any) -> dict[str, Any]:
        payload = {"model": model, "stream": False, "format": schema, **self._base_payload_options(), "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]}
        if think is not None:
            payload["think"] = think
        response = requests.post(self.chat_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or {}
        content = (message.get("content") or "").strip()
        thinking = (message.get("thinking") or "").strip()
        if not content:
            done_reason = normalize_text(data.get("done_reason"))
            if done_reason == "length":
                raise ValueError(f"Ollama chat hit length limit before final JSON content. done_reason={done_reason}; eval_count={data.get('eval_count')}; prompt_eval_count={data.get('prompt_eval_count')}")
            if thinking:
                raise ValueError(f"Ollama chat returned thinking trace but no final JSON content. done_reason={done_reason}; eval_count={data.get('eval_count')}")
            raise ValueError(f"Empty response received from /api/chat: {data}")
        return safe_json_loads(content)

    def _generate_json(self, model: str, system_prompt: str, user_prompt: str, schema: dict[str, Any], think: Any) -> dict[str, Any]:
        payload = {"model": model, "stream": False, "format": schema, **self._base_payload_options(), "system": system_prompt, "prompt": user_prompt}
        if think is not None:
            payload["think"] = think
        response = requests.post(self.generate_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        content = (data.get("response") or "").strip()
        if not content:
            done_reason = normalize_text(data.get("done_reason"))
            if done_reason == "length":
                raise ValueError(f"Ollama generate hit length limit before final JSON content. done_reason={done_reason}; eval_count={data.get('eval_count')}; prompt_eval_count={data.get('prompt_eval_count')}")
            raise ValueError(f"Empty response received from /api/generate: {data}")
        return safe_json_loads(content)

    def complete_json(self, user_prompt: str, system_prompt: str, schema: dict[str, Any], model: Optional[str] = None, think_override: Any = None) -> dict[str, Any]:
        selected_model = model or self.default_model
        if not selected_model:
            raise ValueError("Model name must be provided.")
        resolved_mode, resolved_think = self._resolve_runtime(selected_model)
        if think_override is not None:
            resolved_think = think_override
        if resolved_mode == "chat":
            return self._chat_json(selected_model, system_prompt, user_prompt, schema, think=resolved_think)
        if resolved_mode == "generate":
            return self._generate_json(selected_model, system_prompt, user_prompt, schema, think=resolved_think)
        raise ValueError(f"Unsupported Ollama mode: {resolved_mode}")


_GLOBAL_VLLM_SESSION: requests.Session | None = None
_GLOBAL_VLLM_SESSION_LOCK = threading.Lock()
_GLOBAL_VLLM_REQUEST_SEMAPHORE = threading.BoundedSemaphore(max(1, int(config.get("VLLM_MAX_PARALLEL_REQUESTS") or 6)))


def get_shared_vllm_session() -> requests.Session:
    global _GLOBAL_VLLM_SESSION
    if _GLOBAL_VLLM_SESSION is None:
        with _GLOBAL_VLLM_SESSION_LOCK:
            if _GLOBAL_VLLM_SESSION is None:
                _GLOBAL_VLLM_SESSION = requests.Session()
    return _GLOBAL_VLLM_SESSION


class VLLMChatClient:
    """OpenAI-compatible vLLM client with the same complete_json interface as Ollama client."""

    def __init__(self, base_url: str, api_key: str, timeout: int = 900, default_model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 1024, think: Any = "auto") -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.api_key = api_key
        self.timeout = timeout
        self.default_model = default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.think = think
        self.session = get_shared_vllm_session()

    def _resolve_runtime(self, model: str) -> tuple[str, Any]:
        _ = model
        requested_think = self.think
        if isinstance(requested_think, str) and requested_think.strip().lower() == "auto":
            requested_think = None
        return ("chat", requested_think)

    def complete_json(
            self,
            user_prompt: str,
            system_prompt: str,
            schema: dict[str, Any],
            model: Optional[str] = None,
            think_override: Any = None,
    ) -> dict[str, Any]:
        selected_model = model or self.default_model
        if not selected_model:
            raise ValueError("Model name must be provided.")

        _, resolved_think = self._resolve_runtime(selected_model)
        if think_override is not None:
            resolved_think = think_override

        payload: dict[str, Any] = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": schema,
                    "strict": True,
                },
            },
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        if resolved_think is not None and str(config.get("VLLM_ENABLE_THINK_PARAM")).lower() in {"1", "true", "yes",
                                                                                                 "on"}:
            payload[str(config.get("VLLM_THINK_FIELD_NAME") or "think")] = resolved_think

        # When thinking is explicitly disabled (think_override=False), inject model-specific
        # signals so the vLLM backend actually suppresses the CoT reasoning trace.
        if think_override is False:
            model_lc = normalize_text(selected_model).lower()
            if model_lc.startswith("gpt-oss"):
                # gpt-oss family uses OpenAI-style reasoning_effort; "low" is the minimum
                # that still produces valid output — "none" is not supported by all versions.
                payload["reasoning_effort"] = "low"
            else:
                # Qwen3 / DeepSeek-R1 and similar: disable via chat_template_kwargs.
                payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = False

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with _GLOBAL_VLLM_REQUEST_SEMAPHORE:
            response = self.session.post(
                self.chat_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )

        response.raise_for_status()
        data = response.json()

        choices = data.get("choices") or []
        if not choices:
            raise VLLMEmptyContentError(f"vLLM returned no choices: {data}")

        choice = choices[0]
        message = choice.get("message") or {}

        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
        content = (content or "").strip()

        finish_reason = normalize_text(choice.get("finish_reason")).lower()
        reasoning = normalize_text(
            message.get("reasoning")
            or message.get("reasoning_content")
        )

        if not content:
            if finish_reason == "length" and reasoning:
                raise VLLMTruncatedReasoningError(
                    f"vLLM hit length limit before final content. "
                    f"finish_reason={finish_reason}; usage={data.get('usage')}"
                )
            if reasoning:
                raise VLLMEmptyContentError(
                    f"vLLM returned reasoning but no final content. "
                    f"finish_reason={finish_reason}; usage={data.get('usage')}"
                )
            raise VLLMEmptyContentError(
                f"vLLM returned empty assistant content: {data}"
            )

        return safe_json_loads(content)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def build_llm_client(*, backend: str, timeout: int, default_model: Optional[str], temperature: float, num_ctx: int, num_predict: int, think: Any, runtime_presets: Optional[dict[str, dict[str, Any]]] = None):
    _ = (num_ctx, runtime_presets)
    backend_norm = normalize_text(backend).lower()
    if backend_norm == "vllm":
        return VLLMChatClient(
            base_url=config.get("VLLM_BASE_URL"),
            api_key=config.get("VLLM_API_KEY"),
            timeout=timeout,
            default_model=default_model,
            temperature=temperature,
            max_tokens=num_predict,
            think=think,
        )
    return OllamaLLMClient(
        base_url=config.get("OLLAMA_BASE_URL"),
        mode=config.get("LLM_API_MODE"),
        timeout=timeout,
        default_model=default_model,
        keep_alive=config.get("LLM_KEEP_ALIVE"),
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
        think=think,
        runtime_presets=json.loads(config.get("LLM_MODEL_RUNTIME_PRESETS")),
    )


vectorizer = VectorizerClient(
    url=config.get("VECTORIZER_URL"),
    model=config.get("VECTORIZER_MODEL"),
    client_cert_path=config.get("CLIENT_CERT_PATH"),
    client_key_path=config.get("CLIENT_KEY_PATH"),
    ca_cert_path=config.get("CA_CERT_PATH"),
    timeout=int(config.get("REQUEST_TIMEOUT_EMBED")),
)

llm_client = build_llm_client(
    backend=config.get("LLM_BACKEND"),
    timeout=int(config.get("REQUEST_TIMEOUT_CHAT")),
    default_model=config.get("GENERATE_MODEL"),
    temperature=float(config.get("LLM_TEMPERATURE")),
    num_ctx=int(config.get("LLM_NUM_CTX")),
    num_predict=int(config.get("LLM_NUM_PREDICT")),
    think=config.get("LLM_THINK"),
    runtime_presets=json.loads(config.get("LLM_MODEL_RUNTIME_PRESETS")),
)

not_allowed_rerank_llm_client = build_llm_client(
    backend=config.get("LLM_BACKEND"),
    timeout=int(config.get("REQUEST_TIMEOUT_CHAT")),
    default_model=config.get("GENERATE_MODEL"),
    temperature=float(config.get("LLM_TEMPERATURE")),
    num_ctx=int(config.get("NOT_ALLOWED_LLM_RERANK_NUM_CTX")),
    num_predict=int(config.get("NOT_ALLOWED_LLM_RERANK_NUM_PREDICT")),
    think=parse_think_value(config.get("NOT_ALLOWED_LLM_RERANK_THINK")),
    runtime_presets=json.loads(config.get("LLM_MODEL_RUNTIME_PRESETS")),
)

# Backward-compatible aliases used in business layers.
ollama = llm_client
not_allowed_rerank_ollama = not_allowed_rerank_llm_client
