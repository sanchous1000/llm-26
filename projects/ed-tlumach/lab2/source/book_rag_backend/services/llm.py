import httpx
from typing import List, Optional


class LLMService:
    def __init__(
            self,
            base_url: str,
            api_key: str,
            model: str = "deepseek-chat",
            timeout: int = 600,
            max_tokens: int = 512,
    ):
        self.base_url = base_url.strip().rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def generate(
            self,
            messages: List[dict],
            temperature: float = 0.0,
    ) -> str:
        client = self._get_client()

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }

        response = await client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]
