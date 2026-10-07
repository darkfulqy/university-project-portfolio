import json
import ssl
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import network_route


class NetworkRouteTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.config = Path(self.directory.name) / "route.json"
        self.write_config()

    def tearDown(self):
        self.directory.cleanup()

    def write_config(self, **extra):
        value = {"enabled": True, "proxy": "http://127.0.0.1:17897", "activate_when_listening": True,
                 "hosts": ["gdgpo.czt.gd.gov.cn"]}
        value.update(extra)
        self.config.write_text(json.dumps(value), encoding="utf-8")

    def test_exact_host_only(self):
        with patch.object(network_route, "_listening", return_value=True):
            self.assertEqual(network_route.route_proxy("https://gdgpo.czt.gd.gov.cn/", config_path=self.config), "http://127.0.0.1:17897")
            self.assertIsNone(network_route.route_proxy("https://gdgpo.czt.gd.gov.cn.example.com/", config_path=self.config))
            self.assertIsNone(network_route.route_proxy("https://example.com/", config_path=self.config))

    def test_waits_for_manual_clash_restart(self):
        with patch.object(network_route, "_listening", return_value=False):
            self.assertIsNone(network_route.route_proxy("https://gdgpo.czt.gd.gov.cn/", config_path=self.config))
        with patch.object(network_route, "_listening", return_value=True):
            self.assertIsNotNone(network_route.route_proxy("https://gdgpo.czt.gd.gov.cn/", config_path=self.config))

    def test_disabled_and_nonlocal_proxy(self):
        self.write_config(enabled=False)
        self.assertIsNone(network_route.route_proxy("https://gdgpo.czt.gd.gov.cn/", config_path=self.config))
        self.write_config(proxy="http://example.com:17897")
        with self.assertRaises(ValueError):
            network_route.route_proxy("https://gdgpo.czt.gd.gov.cn/", config_path=self.config)

    def test_python_request_reaches_local_proxy(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                received.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"local proxy response")
            def log_message(self, *_):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        self.write_config(proxy=f"http://127.0.0.1:{server.server_port}")
        try:
            with patch.object(network_route, "CONFIG_PATH", self.config), patch("urllib.request.proxy_bypass", return_value=False):
                req = urllib.request.Request("http://gdgpo.czt.gd.gov.cn/local-test")
                with network_route.open_url(req, timeout=3, context=ssl.create_default_context()) as response:
                    self.assertEqual(response.read(), b"local proxy response")
                self.assertEqual(network_route.curl_route_args(req.full_url), ["--proxy", f"http://127.0.0.1:{server.server_port}", "--noproxy", ""])
            self.assertEqual(received, ["http://gdgpo.czt.gd.gov.cn/local-test"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
