import http.server
import threading

import pytest


@pytest.fixture
def local_http_server(tmp_path):
    served_dir = tmp_path / "served"
    served_dir.mkdir()

    def handler(*args):
        return http.server.SimpleHTTPRequestHandler(*args, directory=str(served_dir))

    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        yield served_dir, f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join()
