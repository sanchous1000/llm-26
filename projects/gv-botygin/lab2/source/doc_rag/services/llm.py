import httpx

from doc_rag.config import LLMConfig


class LLMService:
    """Клиент OpenAI-совместимого chat/completions"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout,
            headers={"Authorization": f"Bearer {config.api_key}"},
        )

    def close(self) -> None:
        self._client.close()

    def generate(self, messages: list[dict]) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort

        response = self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        content = (response.json()["choices"][0]["message"].get("content") or "").strip()
        if not content:
            raise RuntimeError("LLM вернула пустой ответ (возможно, max_tokens ушли на рассуждения)")
        return content
