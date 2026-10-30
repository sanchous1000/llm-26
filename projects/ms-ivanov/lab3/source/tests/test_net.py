import json

import pytest
import requests

from source.net import FetchError, HttpClient
from source.net.validators import (
    require_html,
    require_latexml_html,
    require_pdf,
    require_text,
)


class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None, url="https://h.test/x"):
        self.status_code = status
        self.ok = status < 400
        self._body = body
        self.headers = headers or {}
        self.url = url

    def iter_content(self, size):
        yield self._body

    def json(self):
        return json.loads(self._body)

    def close(self):
        pass


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _client(*responses, delays=None):
    clock = Clock()
    session = FakeSession(*responses)
    client = HttpClient(
        host_delays=delays or {"h.test": 5.0},
        session=session,
        sleep=clock.sleep,
        clock=clock.time,
    )
    return client, session, clock


def test_fetch_file_writes_validated_body_atomically(tmp_path):
    client, _, _ = _client(FakeResponse(body=b"%PDF-data"))
    dest = tmp_path / "d" / "paper.pdf"

    fetched = client.fetch_file("https://h.test/a", dest, require_pdf)

    assert dest.read_bytes() == b"%PDF-data"
    assert fetched.size == 9 and len(fetched.sha256) == 64
    assert not list(dest.parent.glob("*.part"))


def test_invalid_body_leaves_no_file_and_is_permanent(tmp_path):
    client, _, _ = _client(FakeResponse(body=b"<html>login</html>"))
    dest = tmp_path / "paper.pdf"

    with pytest.raises(FetchError) as error:
        client.fetch_file("https://h.test/a", dest, require_pdf)

    assert error.value.permanent
    assert not dest.exists() and not list(tmp_path.glob("*.part"))


def test_not_found_is_permanent_and_not_retried():
    client, session, _ = _client(FakeResponse(status=404))

    with pytest.raises(FetchError) as error:
        client.get_json("https://h.test/a")

    assert error.value.permanent and error.value.status == 404
    assert len(session.calls) == 1


def test_server_errors_are_retried_with_backoff():
    client, session, clock = _client(
        FakeResponse(status=503), FakeResponse(status=502), FakeResponse(body=b"{}")
    )

    assert client.get_json("https://h.test/a") == {}
    assert len(session.calls) == 3
    assert any(sleep >= 2.0 for sleep in clock.sleeps)


def test_exhausted_retries_raise_transient_error():
    client, _, _ = _client(*[requests.ConnectionError("boom")] * 4)

    with pytest.raises(FetchError) as error:
        client.get_json("https://h.test/a")

    assert not error.value.permanent


def test_rate_limited_forbidden_is_transient():
    limited = FakeResponse(status=403, headers={"X-RateLimit-Remaining": "0"})
    client, _, _ = _client(limited, limited, limited, limited)

    with pytest.raises(FetchError) as error:
        client.get_json("https://h.test/a")

    assert not error.value.permanent


def test_requests_to_one_host_are_spaced_by_its_delay():
    client, _, clock = _client(FakeResponse(body=b"{}"), FakeResponse(body=b"{}"))

    client.get_json("https://h.test/a")
    client.get_json("https://h.test/b")

    assert sum(clock.sleeps) == pytest.approx(5.0)


def test_other_hosts_are_not_delayed():
    client, _, clock = _client(FakeResponse(body=b"{}"), FakeResponse(body=b"{}"))

    client.get_json("https://h.test/a")
    client.get_json("https://other.test/b")

    assert clock.sleeps == []


def test_invalid_json_is_a_permanent_error():
    client, _, _ = _client(FakeResponse(body=b"not json"))

    with pytest.raises(FetchError) as error:
        client.get_json("https://h.test/a")

    assert error.value.permanent


@pytest.mark.parametrize(
    ("validator", "body", "valid"),
    [
        (require_pdf, b"%PDF-1.4", True),
        (require_pdf, b"<html>", False),
        (require_html, b"<!DOCTYPE html><p>x", True),
        (require_html, b"%PDF", False),
        (require_latexml_html, b"<html>" + b'<p class="ltx_p">x</p>' * 10, True),
        (
            require_latexml_html,
            b"<html><div class='ltx_page'>No content available",
            False,
        ),
        (require_latexml_html, b"<html>plain page</html>", False),
        (require_text, b"# readme", True),
        (require_text, b"", False),
    ],
)
def test_validators(tmp_path, validator, body, valid):
    path = tmp_path / "f"
    path.write_bytes(body)

    if valid:
        validator(path, "x")
    else:
        with pytest.raises(FetchError):
            validator(path, "x")
