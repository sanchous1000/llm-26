"""HTTP client with per-host throttling, retries and atomic downloads."""

import hashlib
import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.parse import urlparse

import requests

from source import config
from source.net.errors import FetchError
from source.net.models import FetchedFile
from source.net.validators import Validator

logger = logging.getLogger(__name__)

_RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_CHUNK_BYTES = 1 << 16


class HttpClient:
    """Sequential client that never hits one host faster than allowed."""

    def __init__(
        self,
        *,
        host_delays: Mapping[str, float] | None = None,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._host_delays = dict(host_delays or config.HOST_DELAYS_S)
        self._session = session or self._build_session()
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: dict[str, float] = {}

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        session.headers["User-Agent"] = config.USER_AGENT
        return session

    def get_json(self, url: str, headers: Mapping[str, str] | None = None) -> dict:
        """GET a JSON document."""
        response = self._request(url, headers=headers, stream=False)
        try:
            return response.json()
        except ValueError as error:
            raise FetchError(f"invalid JSON from {url}", permanent=True) from error

    def fetch_file(
        self,
        url: str,
        dest: Path,
        validator: Validator,
        headers: Mapping[str, str] | None = None,
    ) -> FetchedFile:
        """Download ``url`` to ``dest``; ``dest`` only appears if the body is valid."""
        response = self._request(url, headers=headers, stream=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(dest.name + ".part")
        content_type = response.headers.get("Content-Type", "")
        try:
            digest, size = self._stream_to_disk(response, partial)
            validator(partial, content_type)
            partial.replace(dest)
        except (OSError, requests.RequestException) as error:
            raise FetchError(
                f"download of {url} failed: {error}", permanent=False
            ) from error
        finally:
            partial.unlink(missing_ok=True)
            response.close()
            self._last_request_at[urlparse(url).netloc.lower()] = self._clock()
        return FetchedFile(dest, response.url, size, digest, content_type)

    @staticmethod
    def _stream_to_disk(response: requests.Response, target: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        with target.open("wb") as handle:
            for chunk in response.iter_content(_CHUNK_BYTES):
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        return digest.hexdigest(), size

    def _request(
        self, url: str, *, headers: Mapping[str, str] | None, stream: bool
    ) -> requests.Response:
        last_error: FetchError | None = None
        for attempt in range(config.MAX_RETRIES + 1):
            if attempt:
                self._sleep(config.BACKOFF_BASE_S * 2 ** (attempt - 1))
            try:
                response = self._send(url, headers, stream)
            except requests.RequestException as error:
                last_error = FetchError(f"{url}: {error}", permanent=False)
                continue
            if response.ok:
                return response
            last_error = self._error_for(url, response)
            response.close()
            if last_error.permanent:
                break
            self._respect_retry_after(response)
        assert last_error is not None
        raise last_error

    def _send(
        self, url: str, headers: Mapping[str, str] | None, stream: bool
    ) -> requests.Response:
        self._throttle(urlparse(url).netloc.lower())
        logger.debug("GET %s", url)
        try:
            return self._session.get(
                url,
                headers=dict(headers or {}),
                stream=stream,
                timeout=config.REQUEST_TIMEOUT_S,
                allow_redirects=True,
            )
        finally:
            self._last_request_at[urlparse(url).netloc.lower()] = self._clock()

    def _throttle(self, host: str) -> None:
        delay = self._host_delays.get(host, config.DEFAULT_HOST_DELAY_S)
        last = self._last_request_at.get(host)
        if last is None:
            return
        wait = delay - (self._clock() - last)
        if wait > 0:
            self._sleep(wait)

    def _respect_retry_after(self, response: requests.Response) -> None:
        retry_after = response.headers.get("Retry-After", "")
        if retry_after.isdigit():
            self._sleep(min(int(retry_after), 120))

    @staticmethod
    def _error_for(url: str, response: requests.Response) -> FetchError:
        status = response.status_code
        rate_limited = response.headers.get("X-RateLimit-Remaining") == "0"
        permanent = (
            400 <= status < 500 and status not in _RETRYABLE_STATUSES
        ) and not rate_limited
        return FetchError(
            f"HTTP {status} for {url}", permanent=permanent, status=status
        )
