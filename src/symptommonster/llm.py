"""A small, backend-agnostic client for locally served language models.

Extraction and tier-3 normalization both go through `LLMClient`, so they are
indifferent to which server answers. The Ollama and MLX backends import their own
dependencies inside `__init__`, which keeps those optional. The JSON helpers pull
structured output back out of free-form model text.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class LLMRequest:
    prompt: str
    system: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    seed: int | None = None


@runtime_checkable
class LLMClient(Protocol):
    def generate(self, request: LLMRequest) -> str:
        """Return the full completion for `request`."""
        ...

    def stream(self, request: LLMRequest) -> Iterator[str]:
        """Yield the completion in pieces as the model produces them."""
        ...


class OllamaBackend:
    """Serve completions from a model running under Ollama.

    Retries with exponential backoff, since a cold model or a busy server will
    occasionally drop the first request.
    """

    def __init__(
        self,
        model: str,
        *,
        host: str | None = None,
        retries: int = 3,
        backoff: float = 2.0,
        options: dict | None = None,
    ) -> None:
        import ollama  # optional dependency, imported on use

        self._client = ollama.Client(host=host) if host else ollama
        self.model = model
        self.retries = retries
        self.backoff = backoff
        self.options = options or {}

    def _options(self, request: LLMRequest) -> dict:
        opts = {"temperature": request.temperature, "num_predict": request.max_tokens}
        if request.seed is not None:
            opts["seed"] = request.seed
        opts.update(self.options)
        return opts

    def _messages(self, request: LLMRequest) -> list[dict]:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        return messages

    def generate(self, request: LLMRequest) -> str:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self._client.chat(
                    model=self.model,
                    messages=self._messages(request),
                    options=self._options(request),
                )
                return response["message"]["content"]
            except Exception as error:  # noqa: BLE001 - retry any transport failure
                last_error = error
                time.sleep(self.backoff * (2**attempt))
        raise RuntimeError("Ollama request failed after retries") from last_error

    def stream(self, request: LLMRequest) -> Iterator[str]:
        chunks = self._client.chat(
            model=self.model,
            messages=self._messages(request),
            options=self._options(request),
            stream=True,
        )
        for chunk in chunks:
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece


class MLXBackend:
    """Serve completions from a model loaded with `mlx_lm`.

    The model and tokenizer load once at construction; the chat template is
    applied per request so system prompts are honored.
    """

    def __init__(self, model: str) -> None:
        from mlx_lm import load  # optional dependency, imported on use

        self.model = model
        self._model, self._tokenizer = load(model)

    def _prompt(self, request: LLMRequest) -> str:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        if self._tokenizer.chat_template is not None:
            return self._tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return request.prompt

    def generate(self, request: LLMRequest) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=request.temperature)
        return generate(
            self._model,
            self._tokenizer,
            prompt=self._prompt(request),
            max_tokens=request.max_tokens,
            sampler=sampler,
            verbose=False,
        )

    def stream(self, request: LLMRequest) -> Iterator[str]:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=request.temperature)
        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt=self._prompt(request),
            max_tokens=request.max_tokens,
            sampler=sampler,
        ):
            yield response.text


def get_client(backend: str, model: str, **kwargs) -> LLMClient:
    """Construct a client by backend name. The model id is an opaque string."""
    if backend == "ollama":
        return OllamaBackend(model, **kwargs)
    if backend == "mlx":
        return MLXBackend(model, **kwargs)
    raise ValueError(f"unknown backend {backend!r}; expected 'ollama' or 'mlx'")


# --- Recovering structured output from free-form model text ----------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _first_balanced(text: str, open_char: str, close_char: str) -> str | None:
    start = text.find(open_char)
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        char = text[i]
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json(text: str) -> Any | None:
    """Return the first JSON value embedded in `text`, or None if there is none."""
    if not text:
        return None
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    try:
        return json.loads(candidate)
    except (ValueError, TypeError):
        pass
    for open_char, close_char in (("{", "}"), ("[", "]")):
        snippet = _first_balanced(candidate, open_char, close_char)
        if snippet is not None:
            try:
                return json.loads(snippet)
            except (ValueError, TypeError):
                continue
    return None


def extract_string_list(text: str, key: str | None = None) -> list[str]:
    """Recover a list of strings from model output.

    Accepts a bare array, an object with `key`, or an object whose first list
    value holds the items. List entries may be strings or small objects keyed by
    `symptom`/`term`/`name`.
    """
    data = extract_json(text)
    if data is None:
        return []
    if isinstance(data, dict):
        if key is not None:
            data = data.get(key, [])
        else:
            data = next((value for value in data.values() if isinstance(value, list)), [])
    if not isinstance(data, list):
        return []

    items: list[str] = []
    for entry in data:
        if isinstance(entry, str):
            items.append(entry.strip())
        elif isinstance(entry, dict):
            value = entry.get("symptom") or entry.get("term") or entry.get("name")
            if isinstance(value, str):
                items.append(value.strip())
    return [item for item in items if item]


__all__ = [
    "LLMClient",
    "LLMRequest",
    "get_client",
    "extract_json",
    "extract_string_list",
]
