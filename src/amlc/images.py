"""Fast, resumable image downloader (past AMLC datasets shipped image URLs, not images).

    from amlc.images import download_images
    manifest = download_images(train["image_link"], IMAGES / "train")
    train["image_path"] = manifest["path"].where(manifest["ok"])

Re-running skips files already on disk, so it is safe to interrupt.
CLI:  uv run python -m amlc.images data/raw/train.csv image_link data/images/train
"""
import hashlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

_local = threading.local()


def url_to_filename(url: str) -> str:
    name = Path(urlparse(url).path).name
    return name if name else hashlib.md5(url.encode()).hexdigest() + ".jpg"


def _session(pool: int) -> requests.Session:
    if not hasattr(_local, "s"):
        s = requests.Session()
        retry = Retry(total=4, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
        s.mount("http://", HTTPAdapter(max_retries=retry, pool_maxsize=pool))
        s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=pool))
        s.headers["User-Agent"] = "Mozilla/5.0"
        _local.s = s
    return _local.s


def _fetch(url: str, dest: Path, timeout: float, pool: int) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        r = _session(pool).get(url, timeout=timeout)
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(r.content)
        tmp.rename(dest)
        return True
    except Exception:
        return False


def download_images(urls, out_dir, n_workers: int = 64, timeout: float = 15.0) -> pd.DataFrame:
    """Download every unique URL into out_dir. Returns a frame aligned with `urls`:
    columns url, path, ok."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    urls = pd.Series(list(urls), dtype="object")
    valid = urls.dropna()
    valid = valid[valid.astype(str).str.startswith("http")]
    unique = valid.unique()
    status = {}
    with ThreadPoolExecutor(n_workers) as ex:
        futs = {ex.submit(_fetch, u, out_dir / url_to_filename(u), timeout, n_workers): u for u in unique}
        for f in tqdm(as_completed(futs), total=len(futs), desc="images"):
            status[futs[f]] = f.result()
    ok = urls.map(lambda u: status.get(u, False))
    paths = urls.map(lambda u: str(out_dir / url_to_filename(u)) if isinstance(u, str) else None)
    n_ok = int(ok.sum())
    print(f"downloaded/present: {n_ok}/{len(urls)}  failed: {len(urls) - n_ok}")
    return pd.DataFrame({"url": urls, "path": paths, "ok": ok})


def find_corrupt(paths, n_workers: int = 16) -> list[str]:
    """Return paths PIL cannot open (truncated downloads, HTML error pages...)."""
    from PIL import Image

    def bad(p):
        try:
            with Image.open(p) as im:
                im.verify()
            return None
        except Exception:
            return p

    with ThreadPoolExecutor(n_workers) as ex:
        return [p for p in tqdm(ex.map(bad, paths), total=len(paths), desc="verify") if p]


if __name__ == "__main__":
    csv_path, col, out = sys.argv[1:4]
    download_images(pd.read_csv(csv_path)[col], out)
