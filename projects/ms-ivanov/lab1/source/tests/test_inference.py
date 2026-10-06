"""Клиент инференса: метрики скорости и тайминги стриминга (локальный стаб-сервер, без модели).

Запуск: python source/main.py test
"""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from llm_lab.inference.client import LMStudioClient
from llm_lab.inference.params import EndpointConfig, GenerationParams
from llm_lab.inference.speed import SpeedInputs, derive_speed_metrics


class SpeedMetricsTest(unittest.TestCase):
    def test_decode_speed_excludes_ttft(self) -> None:
        speed = derive_speed_metrics(SpeedInputs(
            latency_s=11.0, ttft_s=1.0, ttfa_s=1.0,
            completion_tokens=300, answer_chars=900, reasoning_chars=0,
        ))
        self.assertEqual(speed["output_tps"], 30.0)
        self.assertEqual(speed["tps_method"], "decode")
        self.assertEqual(speed["reasoning_tokens_est"], 0)

    def test_unknown_ttft_falls_back_to_end_to_end(self) -> None:
        speed = derive_speed_metrics(SpeedInputs(
            latency_s=4.5, ttft_s=None, ttfa_s=None,
            completion_tokens=256, answer_chars=0, reasoning_chars=982,
        ))
        self.assertAlmostEqual(speed["output_tps"], 56.89, places=2)
        self.assertEqual(speed["tps_method"], "end_to_end")
        self.assertEqual(speed["reasoning_time_share"], 1.0)

    def test_reasoning_tokens_split(self) -> None:
        speed = derive_speed_metrics(SpeedInputs(
            latency_s=60.0, ttft_s=0.3, ttfa_s=58.0,
            completion_tokens=3000, answer_chars=450, reasoning_chars=10000,
        ))
        self.assertEqual(speed["answer_tokens_est"], 150)
        self.assertEqual(speed["reasoning_tokens_est"], 2850)
        self.assertAlmostEqual(speed["reasoning_time_share"], 0.9667, places=3)


class _ReasoningStreamHandler(BaseHTTPRequestHandler):
    """Стрим в стиле LM Studio: сначала reasoning_content, затем content."""

    def log_message(self, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        events = (
            [{"choices": [{"delta": {"reasoning_content": "мысль "}}]}] * 40
            + [{"choices": [{"delta": {"content": "Итого: 1431 ₽"}, "finish_reason": "stop"}]},
               {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 45, "total_tokens": 95}}]
        )
        for event in events:
            self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()
            threading.Event().wait(0.005)
        self.wfile.write(b"data: [DONE]\n\n")


class StreamingTimingIntegrationTest(unittest.TestCase):
    def test_ttft_counts_reasoning_tokens(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), _ReasoningStreamHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = EndpointConfig(base_url=f"http://127.0.0.1:{server.server_port}/v1", timeout_s=10)
            response = LMStudioClient(endpoint).chat(
                "m", [{"role": "user", "content": "x"}], GenerationParams(), stream=True,
            )
        finally:
            server.shutdown()
            server.server_close()

        self.assertTrue(response.ok, response.error)
        self.assertEqual(response.content, "Итого: 1431 ₽")
        self.assertIsNotNone(response.ttft_s)
        self.assertIsNotNone(response.ttfa_s)
        # Первый токен (рассуждение) приходит заметно раньше первого токена ответа.
        self.assertLess(response.ttft_s, response.ttfa_s)
        # Скорость ограничена разумными пределами, а не тысячами ток/с.
        self.assertLess(response.output_tps, 45 / (response.ttfa_s - response.ttft_s) + 1)


if __name__ == "__main__":
    unittest.main()
