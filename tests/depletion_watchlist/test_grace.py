import http.server
import threading

import numpy as np
import xarray as xr

from dark_water.depletion_watchlist.ingest import grace


def test_load_mascons(tmp_path):
    ds = xr.Dataset({"lwe_thickness": (("time", "lat", "lon"), np.zeros((2, 3, 4)))})
    path = tmp_path / "mascons.nc"
    ds.to_netcdf(path)

    loaded = grace.load_mascons(path)

    assert "lwe_thickness" in loaded.data_vars
    assert loaded["lwe_thickness"].shape == (2, 3, 4)


def test_stream_download(tmp_path):
    payload = b"mascon bytes" * 1000
    served_dir = tmp_path / "served"
    served_dir.mkdir()
    (served_dir / "file.nc").write_bytes(payload)

    handler = lambda *args: http.server.SimpleHTTPRequestHandler(
        *args, directory=str(served_dir)
    )
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        dest = grace._stream_download(f"http://127.0.0.1:{port}/file.nc", tmp_path / "dest")
    finally:
        server.shutdown()
        thread.join()

    assert dest.read_bytes() == payload
