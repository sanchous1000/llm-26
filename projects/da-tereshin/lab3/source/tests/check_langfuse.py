"""Проверяет ключи Langfuse, отправку и чтение пробного трейса без LLM и retrieval."""

import json
import os
from pathlib import Path
from time import monotonic, sleep
from uuid import uuid4

from dotenv import load_dotenv
from langfuse import Langfuse, propagate_attributes


def main():
    project = Path(__file__).resolve().parents[2]
    load_dotenv(project / ".env")
    client = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=os.environ["LANGFUSE_BASE_URL"],
        timeout=5,
        sample_rate=1,
    )
    try:
        assert client.auth_check(), "Ключи не прошли проверку"
        session = "lab3-smoke-" + uuid4().hex[:12]
        with client.start_as_current_observation(
            name="lab3-smoke", as_type="span", input={"check": "connection"},
        ) as root:
            with propagate_attributes(
                user_id="da-tereshin", session_id=session, tags=["lab3", "smoke"],
                trace_name="lab3-smoke",
            ):
                with client.start_as_current_observation(
                    name="synthetic-step", as_type="span", input={"value": 1},
                ) as child:
                    child.update(output={"value": 2})
                root.update(output={"status": "ok"})
        client.flush()

        # flush отправляет события, но обработка на сервере может занять время.
        deadline = monotonic() + 30
        while monotonic() < deadline:
            result = client.api.observations.get_many(
                trace_id=root.trace_id, fields="basic,io,trace_context", limit=10,
                request_options={"timeout_in_seconds": 5, "max_retries": 0},
            )
            observations = {item.name: item for item in result.data}
            if {"lab3-smoke", "synthetic-step"} <= observations.keys():
                break
            sleep(1)
        else:
            raise AssertionError("Langfuse не вернул оба span за 30 секунд после отправки")

        saved_root = observations["lab3-smoke"]
        saved_child = observations["synthetic-step"]
        assert saved_root.id == root.id, "ID корневого span отличается"
        assert saved_child.parent_observation_id == root.id, "Нарушена вложенность span"
        assert saved_root.session_id == session and saved_child.session_id == session, "Не сохранён session_id"
        assert saved_root.user_id == "da-tereshin", "Не сохранён user_id"
        assert json.loads(saved_root.input) == {"check": "connection"}, "Не сохранён input корневого span"
        assert json.loads(saved_root.output) == {"status": "ok"}, "Не сохранён output корневого span"
        assert json.loads(saved_child.input) == {"value": 1}, "Не сохранён input вложенного span"
        assert json.loads(saved_child.output) == {"value": 2}, "Не сохранён output вложенного span"
        assert {"lab3", "smoke"} <= set(saved_root.tags or []), "Не сохранены теги"
        print("OK: ключи, приём событий, вложенность, session_id и input/output проверены.")
        print("Session:", session)
        print("Trace:", client.get_trace_url(trace_id=root.trace_id))
    finally:
        client.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Не печатаем содержимое .env, заголовки HTTP и объекты клиента.
        status = getattr(error, "status_code", None)
        detail = str(error) if isinstance(error, AssertionError) else type(error).__name__
        raise SystemExit(
            f"Проверка Langfuse не прошла: {detail}; HTTP={status}. "
            "Проверь запуск сервера и LANGFUSE_BASE_URL/PUBLIC_KEY/SECRET_KEY в lab3/.env."
        ) from None
