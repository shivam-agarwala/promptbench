"""Client for any OpenAI-compatible /chat/completions endpoint.

Ollama, OpenAI, OpenRouter and the Hugging Face router share the wire format; only the
base URL, key env var and model id differ. urllib rather than the openai package, to
keep the project dependency-free.
"""
import json
import os
import random
import time
import urllib.error
import urllib.request

from harness import Completion, notify

# Defaults only. Model availability changes; override with --model.
PROVIDERS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": "OLLAMA_API_KEY",
        "model": "llama3.2",
        "note": "Local Ollama, the default. No account, no key, no quota. "
                "Run `ollama serve` and `ollama pull llama3.2` first.",
        "key_optional": True,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
        "note": "OpenAI directly. Paid.",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "note": "OpenRouter. Some model ids carry a :free suffix.",
    },
    "huggingface": {
        "base_url": "https://router.huggingface.co/v1",
        "key_env": "HF_TOKEN",
        "model": "openai/gpt-oss-20b",
        "note": "Hugging Face Inference Providers router. Monthly credits run out fast.",
    },
}


class OpenAICompatibleClient:
    """Implements the same complete() contract as MockClient."""

    def __init__(self, base_url, model, key_env, max_retries=5, temperature=0.0,
                 cache=None, key_optional=False, timeout=60):
        self.base_url = base_url.rstrip("/")
        self.model, self.key_env = model, key_env
        self.max_retries, self.temperature = max_retries, temperature
        self.cache, self.timeout = cache, timeout
        self.key = os.environ.get(key_env, "")
        if not self.key and not key_optional:
            raise SystemExit(
                f"{key_env} is not set.\n"
                f"  Put it in a .env file next to evaluate.py, or export it.\n"
                f"  Or run with --dry-run, which needs no key.")
        # Some OpenAI-family models require max_completion_tokens instead. Detected on
        # the first 400 and reused for the rest of the run.
        self._token_field = "max_tokens"

    def _post(self, payload):
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _error_text(err):
        """Extract the server's message from an HTTPError body."""
        try:
            detail = json.loads(err.read().decode("utf-8"))
            if isinstance(detail, dict):
                inner = detail.get("error", detail)
                return str(inner.get("message", inner)) if isinstance(inner, dict) else str(inner)
            return str(detail)
        except Exception:
            return ""

    def list_models(self, limit=30):
        """GET {base_url}/models. Best effort; returns [] on any failure."""
        try:
            headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
            req = urllib.request.Request(f"{self.base_url}/models", method="GET",
                                         headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return [m["id"] for m in data.get("data", [])][:limit]
        except Exception:
            return []

    def complete(self, system, user, strategy, message, max_tokens, cache_key=None, stop=()):
        if self.cache and cache_key:
            hit = self.cache.get(cache_key)
            if hit:
                return hit
        t0 = time.perf_counter()
        last_err = ""
        for attempt in range(1, self.max_retries + 1):
            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                self._token_field: max_tokens,
                "temperature": self.temperature,
            }
            if stop:
                payload["stop"] = list(stop)
            try:
                _, data = self._post(payload)
                choice = data["choices"][0]
                text = (choice.get("message") or {}).get("content") or ""
                usage = data.get("usage") or {}
                done = Completion(
                    text=text, latency_ms=(time.perf_counter() - t0) * 1000.0,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0), attempts=attempt)
                if self.cache and cache_key:
                    self.cache.put(cache_key, done)
                return done
            except urllib.error.HTTPError as err:
                detail = self._error_text(err)
                last_err = f"HTTP {err.code}: {detail or err.reason}"
                if err.code == 400 and "max_completion_tokens" in detail \
                        and self._token_field == "max_tokens":
                    self._token_field = "max_completion_tokens"
                    notify("note: endpoint wants max_completion_tokens; switching and retrying.")
                    continue
                if err.code not in (408, 409, 429) and err.code < 500:
                    break  # 401, 403, 404 will fail the same way forever
                if attempt == self.max_retries:
                    break
                after = err.headers.get("retry-after") if err.headers else None
                wait = float(after) if (after or "").replace(".", "", 1).isdigit() \
                    else (2 ** (attempt - 1)) * 0.5
                time.sleep(wait + random.random() * 0.3)
            except (urllib.error.URLError, TimeoutError, OSError) as err:
                last_err = f"{type(err).__name__}: {err}"
                if attempt == self.max_retries:
                    break
                time.sleep((2 ** (attempt - 1)) * 0.5 + random.random() * 0.3)
            except (KeyError, IndexError, ValueError) as err:
                # Response did not match the expected schema. Not retryable.
                last_err = f"unexpected response shape: {type(err).__name__}: {err}"
                break
        return Completion(latency_ms=(time.perf_counter() - t0) * 1000.0,
                          error=last_err, attempts=attempt)


def build_provider_client(name, model=None, base_url=None, key_env=None, **kwargs):
    if name == "custom":
        if not base_url:
            raise SystemExit("--provider custom needs --base-url and --model")
        return OpenAICompatibleClient(base_url, model, key_env or "LLM_API_KEY",
                                      key_optional=not key_env, **kwargs)
    if name not in PROVIDERS:
        raise SystemExit(f"unknown provider {name!r}. Options: {list(PROVIDERS)} or custom")
    cfg = PROVIDERS[name]
    return OpenAICompatibleClient(cfg["base_url"], model or cfg["model"], cfg["key_env"],
                                  key_optional=cfg.get("key_optional", False), **kwargs)
