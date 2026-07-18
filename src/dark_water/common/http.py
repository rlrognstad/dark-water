from pathlib import Path

import requests


def stream_download(url: str, dest_dir: Path) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / url.rsplit("/", 1)[-1]
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with dest_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return dest_path
