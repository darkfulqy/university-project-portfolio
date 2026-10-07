from __future__ import annotations

from dataclasses import dataclass
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class HttpClient:
    user_agent: str
    timeout: int = 30
    min_interval_seconds: float = 0.12

    def __post_init__(self) -> None:
        object.__setattr__(self, "_last_fetch", 0.0)

    def get_text(self, url: str, *, accept: str = "*/*") -> str:
        data = self.get_bytes(url, accept=accept)
        return data.decode("utf-8", errors="replace")

    def get_json(self, url: str) -> object:
        return json.loads(self.get_text(url, accept="application/json"))

    def get_bytes(self, url: str, *, accept: str = "*/*") -> bytes:
        elapsed = time.monotonic() - self._last_fetch
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        headers = {
            "User-Agent": self.user_agent or "ai-stock-discovery-mvp/0.1 no-contact-configured",
            "Accept": accept,
            "Accept-Encoding": "identity",
        }
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                object.__setattr__(self, "_last_fetch", time.monotonic())
                return response.read()
        except HTTPError as exc:
            raise FetchError(f"HTTP {exc.code} while fetching {url}") from exc
        except URLError as exc:
            raise FetchError(f"Network error while fetching {url}: {exc.reason}") from exc
