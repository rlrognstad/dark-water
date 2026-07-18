from dark_water.common.http import stream_download


def test_stream_download(local_http_server, tmp_path):
    served_dir, base_url = local_http_server
    payload = b"mascon bytes" * 1000
    (served_dir / "file.nc").write_bytes(payload)

    dest = stream_download(f"{base_url}/file.nc", tmp_path / "dest")

    assert dest.read_bytes() == payload
