import os
import sys
import time
import json
import zipfile
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UQ_DATASETS = {
    'NF-ToN-IoT-v3': '343e2e8c-6e6e-4a0c-813d-a46acea1b7f4',
    'NF-BoT-IoT-v3': 'db0187cc-1f78-4215-8e34-303364498866',
    'NF-UNSW-NB15-v3': 'abd2f5d8-e268-4ff0-84fb-f2f7b3ca3e8f',
    'NF-CSE-CIC-IDS2018-v3': '4ac221b1-6bd6-42b1-bdf7-03f4fc7efb22'
}

def get_s3_url(uuid):
    url_dl = f'https://api.rdm.uq.edu.au/datasets/{uuid}/download'
    req_dl = urllib.request.Request(url_dl, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req_dl, timeout=15) as r:
        dl = json.loads(r.read().decode('utf-8'))
        return dl.get('url')

def download_chunk(url, start_byte, end_byte, filepath, chunk_id):
    req = urllib.request.Request(url, headers={'Range': f'bytes={start_byte}-{end_byte}', 'User-Agent': 'Mozilla/5.0'})
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                with open(filepath, 'r+b') as f:
                    f.seek(start_byte)
                    f.write(data)
                return len(data)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1 + attempt)

def download_file_parallel(url, dest_path, num_threads=16):
    req = urllib.request.Request(url, headers={'Range': 'bytes=0-0', 'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as resp:
        content_range = resp.headers.get('Content-Range', '')
        total_size = int(content_range.split('/')[-1])
    
    print(f"Downloading {os.path.basename(dest_path)} ({total_size / (1024*1024):.2f} MB) using {num_threads} threads...")
    start_time = time.time()
    
    with open(dest_path, 'wb') as f:
        f.seek(total_size - 1)
        f.write(b'\0')
    
    chunk_size = total_size // num_threads
    futures = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            start_b = i * chunk_size
            end_b = total_size - 1 if i == num_threads - 1 else (i + 1) * chunk_size - 1
            futures.append(executor.submit(download_chunk, url, start_b, end_b, dest_path, i))
        
        for future in futures:
            future.result()
            
    elapsed = time.time() - start_time
    speed = total_size / (1024*1024*elapsed) if elapsed > 0 else 0
    print(f"Completed in {elapsed:.1f}s ({speed:.2f} MB/s)!")

def download_and_extract_uq(name, uuid, data_dir):
    print(f"\n==================== Processing {name} ====================")
    zip_path = os.path.join(data_dir, f'{name}.zip')
    s3_url = get_s3_url(uuid)
    download_file_parallel(s3_url, zip_path, num_threads=16)
    
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        for info in z.infolist():
            print(f"  Extracting: {info.filename} ({info.file_size / (1024*1024):.2f} MB)")
            # Extract flat or into data_dir
            extracted_path = z.extract(info, data_dir)
            # If extracted into a nested subfolder, let's see where it lands
    
    if os.path.exists(zip_path):
        os.remove(zip_path)
    print(f"{name} download and extraction complete!")

if __name__ == '__main__':
    data_dir = os.path.abspath('temporal_benchmark_v3/data')
    os.makedirs(data_dir, exist_ok=True)
    
    target = sys.argv[1] if len(sys.argv) > 1 else 'NF-ToN-IoT-v3'
    if target in UQ_DATASETS:
        download_and_extract_uq(target, UQ_DATASETS[target], data_dir)
    elif target == 'all':
        for name, uuid in UQ_DATASETS.items():
            download_and_extract_uq(name, uuid, data_dir)
    else:
        print(f"Unknown target: {target}. Available: {list(UQ_DATASETS.keys())} or 'all'")
