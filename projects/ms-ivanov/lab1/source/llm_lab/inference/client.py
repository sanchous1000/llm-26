"""Клиент OpenAI-совместимого REST API (LM Studio / vLLM / Ollama / SGLang)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterable

import requests

from llm_lab.evaluation.text import has_visible_answer, split_reasoning
from llm_lab.inference.params import EndpointConfig, GenerationParams
from llm_lab.inference.speed import SpeedInputs, derive_speed_metrics

# Грубая оценка, если сервер не вернул usage: ~3.5 символа на токен для
# смеси русского и латиницы. Такие значения помечаются estimated=True.
CHARS_PER_TOKEN_ESTIMATE = 3.5


class LLMRequestError(RuntimeError):
    """Ошибка обращения к серверу инференса."""


@dataclass(frozen=True)
class ChatResponse:
    """Результат одного запроса вместе с метриками."""

    content: str
    reasoning: str
    raw_content: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tokens_estimated: bool
    latency_s: float
    ttft_s: float | None
    request_payload: dict[str, Any]
    error: str = ""
    # Время до первого токена ВИДИМОГО ответа; у reasoning-моделей ≈ время рассуждений.
    ttfa_s: float | None = None

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def speed(self) -> dict[str, Any]:
        return derive_speed_metrics(SpeedInputs(
            latency_s=self.latency_s,
            ttft_s=self.ttft_s,
            ttfa_s=self.ttfa_s,
            completion_tokens=self.completion_tokens,
            answer_chars=len(self.content),
            reasoning_chars=len(self.reasoning),
        ))

    @property
    def output_tps(self) -> float | None:
        return self.speed["output_tps"]

    @property
    def truncated(self) -> bool:
        return self.finish_reason in {"length", "max_tokens"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "reasoning": self.reasoning,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tokens_estimated": self.tokens_estimated,
            "latency_s": round(self.latency_s, 3),
            **self.speed,
            "truncated": self.truncated,
            "error": self.error,
        }


@dataclass(frozen=True)
class ModelInfo:
    id: str
    state: str = "unknown"
    kind: str = "unknown"


class LMStudioClient:
    """Минимальный клиент чат-комплишенов с измерением метрик."""

    def __init__(self, endpoint: EndpointConfig, include_extra_params: bool = True) -> None:
        self._endpoint = endpoint
        self._include_extra = include_extra_params
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {endpoint.api_key}",
            "Content-Type": "application/json",
        })

    @property
    def base_url(self) -> str:
        return self._endpoint.base_url

    # -- служебные запросы -------------------------------------------------

    def list_models(self) -> tuple[ModelInfo, ...]:
        """Список моделей. Сначала пробуем native-API LM Studio (есть статус)."""
        native = self._try_native_models()
        if native:
            return native
        try:
            response = self._session.get(f"{self.base_url}/models", timeout=30)
            response.raise_for_status()
            data = response.json().get("data", [])
        except (requests.RequestException, ValueError) as exc:
            raise LLMRequestError(f"Не удалось получить список моделей: {exc}") from exc
        return tuple(ModelInfo(id=item.get("id", "")) for item in data if item.get("id"))

    def _try_native_models(self) -> tuple[ModelInfo, ...]:
        native_url = self.base_url.rsplit("/v1", 1)[0] + "/api/v0/models"
        try:
            response = self._session.get(native_url, timeout=10)
            response.raise_for_status()
            data = response.json().get("data", [])
        except (requests.RequestException, ValueError):
            return ()
        return tuple(
            ModelInfo(
                id=item.get("id", ""),
                state=item.get("state", "unknown"),
                kind=item.get("type", "unknown"),
            )
            for item in data
            if item.get("id")
        )

    # -- основной запрос ---------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        params: GenerationParams,
        stream: bool = False,
    ) -> ChatResponse:
        """Один чат-комплишен. Ошибки не бросаются, а возвращаются в ChatResponse."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            **params.to_payload(include_extra=self._include_extra),
        }
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}

        try:
            return self._execute(payload, messages, stream)
        except LLMRequestError as exc:
            if self._include_extra and "top_k" in payload:
                # Сервер может не принимать llama.cpp-специфичные параметры —
                # повторяем без них, явно помечая это в отчёте.
                fallback = {k: v for k, v in payload.items() if k not in ("top_k", "repeat_penalty", "min_p")}
                try:
                    result = self._execute(fallback, messages, stream)
                except LLMRequestError as retry_exc:
                    return self._failed(payload, str(retry_exc))
                return result
            return self._failed(payload, str(exc))

    def _execute(self, payload: dict[str, Any], messages: list[dict[str, str]], stream: bool) -> ChatResponse:
        url = f"{self.base_url}/chat/completions"
        started = time.perf_counter()
        try:
            response = self._session.post(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                timeout=self._endpoint.timeout_s,
                stream=stream,
            )
        except requests.RequestException as exc:
            raise LLMRequestError(f"сеть/таймаут: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text[:500].replace("\n", " ")
            raise LLMRequestError(f"HTTP {response.status_code}: {detail}")

        if stream:
            return self._read_stream(response, payload, messages, started)
        return self._read_single(response, payload, messages, started)

    def _read_single(
        self,
        response: requests.Response,
        payload: dict[str, Any],
        messages: list[dict[str, str]],
        started: float,
    ) -> ChatResponse:
        response.encoding = response.encoding or "utf-8"
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMRequestError(f"невалидный JSON в ответе: {exc}") from exc
        latency = time.perf_counter() - started

        choices = body.get("choices") or []
        if not choices:
            raise LLMRequestError("сервер вернул пустой список choices")
        message = choices[0].get("message") or {}
        raw_content = message.get("content") or ""
        explicit_reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        finish_reason = choices[0].get("finish_reason") or "unknown"

        return self._build(
            raw_content=raw_content,
            explicit_reasoning=explicit_reasoning,
            finish_reason=finish_reason,
            usage=body.get("usage") or {},
            messages=messages,
            payload=payload,
            latency=latency,
            ttft=None,
        )

    def _read_stream(
        self,
        response: requests.Response,
        payload: dict[str, Any],
        messages: list[dict[str, str]],
        started: float,
    ) -> ChatResponse:
        chunks: list[str] = []
        reasoning_chunks: list[str] = []
        finish_reason = "unknown"
        usage: dict[str, Any] = {}
        # TTFT — первый токен ЛЮБОГО вида: рассуждения часто идут отдельным полем
        # reasoning_content, и засекать только content значит измерять время до ответа.
        ttft: float | None = None
        ttfa: float | None = None

        for line in self._iter_sse(response):
            if line == "[DONE]":
                break
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                reasoning_piece = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if (piece or reasoning_piece) and ttft is None:
                    ttft = time.perf_counter() - started
                if piece:
                    chunks.append(piece)
                    # Ответ может начинаться с inline-блока <think> — ждём видимый текст.
                    if ttfa is None and has_visible_answer("".join(chunks)):
                        ttfa = time.perf_counter() - started
                if reasoning_piece:
                    reasoning_chunks.append(reasoning_piece)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

        latency = time.perf_counter() - started
        return self._build(
            raw_content="".join(chunks),
            explicit_reasoning="".join(reasoning_chunks),
            finish_reason=finish_reason,
            usage=usage,
            messages=messages,
            payload=payload,
            latency=latency,
            ttft=ttft,
            ttfa=ttfa,
        )

    @staticmethod
    def _iter_sse(response: requests.Response) -> Iterable[str]:
        """Строки SSE. Декодируем сами: для text/event-stream без charset
        requests подставляет ISO-8859-1 и ломает кириллицу."""
        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line.startswith("data:"):
                yield line[len("data:"):].strip()

    def _build(
        self,
        raw_content: str,
        explicit_reasoning: str,
        finish_reason: str,
        usage: dict[str, Any],
        messages: list[dict[str, str]],
        payload: dict[str, Any],
        latency: float,
        ttft: float | None,
        ttfa: float | None = None,
    ) -> ChatResponse:
        inline_answer, inline_reasoning = split_reasoning(raw_content)
        reasoning = (explicit_reasoning + "\n\n" + inline_reasoning).strip()

        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or 0)
        estimated = False
        if completion_tokens <= 0:
            estimated = True
            prompt_text = "".join(m.get("content", "") for m in messages)
            prompt_tokens = prompt_tokens or _estimate_tokens(prompt_text)
            completion_tokens = _estimate_tokens(raw_content + reasoning)
            total_tokens = prompt_tokens + completion_tokens

        return ChatResponse(
            content=inline_answer,
            reasoning=reasoning,
            raw_content=raw_content,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens or (prompt_tokens + completion_tokens),
            tokens_estimated=estimated,
            latency_s=latency,
            ttft_s=ttft,
            ttfa_s=ttfa,
            request_payload=_payload_without_messages(payload),
        )

    @staticmethod
    def _failed(payload: dict[str, Any], error: str) -> ChatResponse:
        return ChatResponse(
            content="",
            reasoning="",
            raw_content="",
            finish_reason="error",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            tokens_estimated=True,
            latency_s=0.0,
            ttft_s=None,
            request_payload=_payload_without_messages(payload),
            error=error,
        )


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / CHARS_PER_TOKEN_ESTIMATE))


def _payload_without_messages(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k != "messages"}
