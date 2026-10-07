"""Use a dedicated loopback DIRECT listener for selected procurement hosts."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name("network_route.json")
_probe_lock = threading.Lock()
_probe_cache = {}


def _listening(host, port):
    key = (host, port)
    with _probe_lock:
        saved = _probe_cache.get(key)
        if saved and time.monotonic() - saved[0] < 5:
            return saved[1]
        try:
            with socket.create_connection(key, timeout=0.25):
                reachable = True
        except OSError:
            reachable = False
        _probe_cache[key] = (time.monotonic(), reachable)
        return reachable


def route_proxy(url, *, config_path=None):
    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        return None
    config = json.loads(path.read_text(encoding="utf-8"))
    if not config.get("enabled"):
        return None
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    if host not in {str(x).lower() for x in config.get("hosts", [])}:
        return None
    proxy = urllib.parse.urlsplit(config["proxy"])
    if proxy.scheme != "http" or proxy.hostname not in ("127.0.0.1", "localhost", "::1") or not proxy.port or proxy.username or proxy.password:
        raise ValueError("采购直连入口必须是无账号的本机HTTP端口")
    if config.get("activate_when_listening", True) and not _listening(proxy.hostname, proxy.port):
        return None
    return config["proxy"]


def open_url(request, *, timeout, context):
    url = request.full_url if isinstance(request, urllib.request.Request) else str(request)
    proxy = route_proxy(url)
    if not proxy:
        return urllib.request.urlopen(request, timeout=timeout, context=context)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
        urllib.request.HTTPSHandler(context=context),
    )
    return opener.open(request, timeout=timeout)


def curl_route_args(url):
    proxy = route_proxy(url)
    return ["--proxy", proxy, "--noproxy", ""] if proxy else []


if __name__ == "__main__":
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    print(json.dumps({"proxy": config["proxy"], "routes": {
        host: "DIRECT入口已就绪" if route_proxy("https://" + host) else "沿用原网络，等待Clash加载直连入口"
        for host in config["hosts"]
    }}, ensure_ascii=False, indent=2))
