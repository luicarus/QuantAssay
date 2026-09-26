"""Fixed serving request set (mvp-prd.md §4).

The workload must be reproducible and provably identical on both sides: every
request carries a stable id and a hash of its normalized prompt text. The set is
materialized from a tokenizer only to hit the input-token target — the *texts*
are fixed here, never generated per run.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantassay.contracts import sha256_of

#: The built-in request set. Deliberately short, distinct topics so a truncated
#: or crossed-up response is visible in the transcript rather than hidden by
#: near-identical outputs. Temperature is 0, so outputs are deterministic.
REQUEST_TEXTS: tuple[tuple[str, str], ...] = (
    ("req-01", "Explain in one paragraph why the sky appears blue during the day."),
    ("req-02", "List three differences between a compiler and an interpreter."),
    ("req-03", "Summarize what a database index does and when it can hurt."),
    ("req-04", "Describe the water cycle in four sentences."),
    ("req-05", "What is the difference between latency and throughput?"),
    ("req-06", "Explain why hash tables have average constant lookup time."),
    ("req-07", "Give a short definition of idempotency in HTTP."),
    ("req-08", "Describe how a transformer attends to previous tokens."),
    ("req-09", "Why does quantization reduce memory usage in neural networks?"),
    ("req-10", "Name two reasons a service might return a 503 error."),
    ("req-11", "Explain the purpose of a warm-up period in benchmarking."),
    ("req-12", "What does a p95 latency tell you that an average does not?"),
    ("req-13", "Describe the difference between symmetric and asymmetric keys."),
    ("req-14", "Why can caching make a slow system faster but less correct?"),
    ("req-15", "Explain what a KV cache stores during text generation."),
    ("req-16", "Give one advantage of batching requests on a GPU."),
    ("req-17", "What is the role of a group size in weight quantization?"),
    ("req-18", "Describe how a tokenizer splits text into tokens."),
    ("req-19", "Why is measuring the first token differently from the rest useful?"),
    ("req-20", "Summarize the tradeoff between model size and inference speed."),
)


@dataclass(frozen=True)
class Request:
    """One immutable serving request."""

    request_id: str
    text: str

    @property
    def text_sha256(self) -> str:
        return sha256_of({"text": self.text})


def build_request_set(limit: int | None = None) -> list[Request]:
    """The fixed request set, optionally truncated to the first ``limit`` items."""
    texts = REQUEST_TEXTS if limit is None else REQUEST_TEXTS[:limit]
    if not texts:
        raise ValueError("request set would be empty")
    return [Request(request_id=rid, text=text) for rid, text in texts]


def request_set_hashes(requests: list[Request]) -> list[str]:
    """Hash list in the order the requests will be sent (order is part of the workload)."""
    return [request.text_sha256 for request in requests]


def chat_messages(request: Request) -> list[dict[str, str]]:
    """The single-turn chat payload shared by both sides."""
    return [{"role": "user", "content": request.text}]


def chat_payload(
    request: Request,
    *,
    model_id: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    thinking_disabled: bool = True,
) -> dict[str, object]:
    """OpenAI-compatible streaming payload for one request.

    ``enable_thinking=False`` is passed explicitly: Qwen3 defaults to thinking
    on, which would change output length and make the two sides incomparable.
    """
    payload: dict[str, object] = {
        "model": model_id,
        "messages": chat_messages(request),
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if thinking_disabled:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload
