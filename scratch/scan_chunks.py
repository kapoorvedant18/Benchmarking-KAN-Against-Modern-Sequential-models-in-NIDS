import pandas as pd

csv_path = 'temporal_benchmark_v3/data/NF-BoT-IoT-v3.csv'
chunk_size = 500_000

print(f"{'Chunk':<6} | {'Row Range':<22} | {'Test Split (Last 15% / ~75k flows)'}")
print('-' * 80)

for idx, chunk in enumerate(pd.read_csv(csv_path, usecols=['Attack'], chunksize=chunk_size)):
    n = len(chunk)
    test_slice = chunk.iloc[int(0.85 * n):]
    vc = dict(test_slice['Attack'].value_counts())
    start_r = idx * chunk_size
    end_r = start_r + n
    print(f"Chunk {idx:<2} | {start_r:>9,} - {end_r:>9,} | {vc}")
