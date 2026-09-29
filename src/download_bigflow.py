"""
Downloader and extractor for BigFlow-NIDS.
Downloads BigFlow-NIDS-CSV.zip from the direct S3 endpoint and extracts it to temporal_benchmark_v3/data/.
"""
import sys
import os
import time
import zipfile
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BIGFLOW_URL = "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/4db3ab96-bdd0-480f-bc42-efc24e3198b3"
TARGET_DIR = Path(__file__).resolve().parent.parent / 'data'
ZIP_PATH = TARGET_DIR / 'BigFlow-NIDS-CSV.zip'
CSV_PATH = TARGET_DIR / 'BigFlow-NIDS.csv'


def download_bigflow(max_workers: int = 16, chunk_size: int = 8 * 1024 * 1024):
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 10_000_000:
        print(f"BigFlow-NIDS.csv already exists ({CSV_PATH.stat().st_size / (1024*1024):.1f} MB). Skipping download.", flush=True)
        return

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    r = requests.head(BIGFLOW_URL, headers=headers, timeout=30)
    total_size = int(r.headers.get('content-length', 0))
    print(f"Starting BigFlow-NIDS download ({total_size / (1024*1024):.1f} MB)...", flush=True)

    with open(ZIP_PATH, "wb") as f:
        f.seek(total_size - 1)
        f.write(b"\0")

    ranges = []
    for start in range(0, total_size, chunk_size):
        end = min(start + chunk_size - 1, total_size - 1)
        ranges.append((start, end))

    start_time = time.time()
    completed_bytes = 0

    def download_range(byte_range):
        r_start, r_end = byte_range
        h = {'Range': f'bytes={r_start}-{r_end}', 'User-Agent': 'Mozilla/5.0'}
        for attempt in range(5):
            try:
                resp = requests.get(BIGFLOW_URL, headers=h, timeout=40)
                if resp.status_code in [200, 206]:
                    with open(ZIP_PATH, "r+b") as fp:
                        fp.seek(r_start)
                        fp.write(resp.content)
                    return len(resp.content)
            except Exception:
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"Failed chunk {r_start}-{r_end}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_range, rng): rng for rng in ranges}
        for future in as_completed(futures):
            nbytes = future.result()
            completed_bytes += nbytes
            pct = (completed_bytes / total_size) * 100
            elapsed = time.time() - start_time
            speed = (completed_bytes / (1024 * 1024)) / max(0.1, elapsed)
            print(f"\r  [BigFlow-NIDS] {completed_bytes / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({pct:.1f}%) | {speed:.2f} MB/s", end="", flush=True)

    print(f"\nDownload completed in {time.time() - start_time:.1f}s. Extracting...", flush=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(TARGET_DIR)
        extracted = z.namelist()
    print(f"Extracted: {extracted}", flush=True)
    
    # Locate extracted CSV and rename/ensure BigFlow-NIDS.csv
    candidates = list(TARGET_DIR.glob("*BigFlow*.csv")) + [TARGET_DIR / n for n in extracted if n.endswith('.csv')]
    for c in candidates:
        if c.exists() and c != CSV_PATH:
            print(f"Renaming {c.name} to {CSV_PATH.name}...", flush=True)
            c.replace(CSV_PATH)
            break

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    print("BigFlow-NIDS ready.", flush=True)


if __name__ == '__main__':
    download_bigflow()
