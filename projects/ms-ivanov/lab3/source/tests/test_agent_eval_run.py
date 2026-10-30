import asyncio

from source.agent.evaluation import run as eval_run
from source.agent.evaluation.dataset import load_tasks
from source.agent.opencode import proxy_server
from source.agent.tracing import Tracing


class FakeLangfuse:
    def __init__(self):
        self.datasets, self.items, self.flushed = [], [], False

    def create_dataset(self, **kwargs):
        self.datasets.append(kwargs)

    def create_dataset_item(self, **kwargs):
        self.items.append(kwargs)

    def flush(self):
        self.flushed = True


def test_upload_creates_one_item_per_task_with_task_ids():
    client = FakeLangfuse()
    tasks = load_tasks()

    count = eval_run.upload_dataset(Tracing(client), tasks)

    assert count == len(tasks) == len(client.items)
    assert client.datasets[0]["name"] == eval_run.DATASET_NAME
    first = client.items[0]
    assert first["id"] == tasks[0].id and first["metadata"]["task_id"] == tasks[0].id
    assert first["input"]["turns"][0] == tasks[0].turns[0].user
    assert client.flushed


def test_records_keep_the_last_run_of_a_task(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_run, "RUNS_DIR", tmp_path)
    writer = eval_run._RecordWriter(tmp_path / "r.jsonl")
    writer.write({"task_id": "S1", "turns": [], "attempt": 1})
    writer.write({"task_id": "S2", "turns": []})
    writer.write({"task_id": "S1", "turns": [], "attempt": 2})

    records = eval_run.load_records("r")

    assert [r["task_id"] for r in records] == ["S1", "S2"]
    assert records[0]["attempt"] == 2
    assert len((tmp_path / "r.jsonl").read_text().splitlines()) == 3


def test_tracing_is_required_for_runs():
    try:
        eval_run._client(Tracing())
    except RuntimeError as error:
        assert "Langfuse" in str(error)
    else:
        raise AssertionError("expected an error without Langfuse")
    assert "S01-gangbo-mccann-assumptions" in eval_run.tasks_by_id()


def test_proxy_check_is_false_when_nothing_listens():
    assert not proxy_server.is_up("http://127.0.0.1:1/health")
    assert not asyncio.run(proxy_server.is_up_async("http://127.0.0.1:1/health"))


def test_judged_file_stands_in_for_missing_raw_records(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_run, "RUNS_DIR", tmp_path)
    judged = [
        {"task_id": "S1", "turns": [{"turn": 0, "judge": {"verdict": "correct"}}]}
    ]

    eval_run.save_judged("r", judged)

    assert eval_run.load_records("r") == judged == eval_run.load_judged("r")
    assert eval_run.load_meta("r") == {}
