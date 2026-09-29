"""
Configuration module for Temporal NetFlow Benchmark v3.
All paths are strictly self-contained within temporal_benchmark_v3/.
"""
import os
from pathlib import Path
import torch

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BENCHMARK_ROOT / 'data'
RESULTS_DIR = BENCHMARK_ROOT / 'results'
CHECKPOINTS_DIR = BENCHMARK_ROOT / 'checkpoints'
FIGURES_DIR = RESULTS_DIR / 'figures'
TABLES_DIR = RESULTS_DIR / 'tables'

for d in [DATA_DIR, RESULTS_DIR, CHECKPOINTS_DIR, FIGURES_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

DATASETS = [
    'NF-ToN-IoT-v3',
    'NF-BoT-IoT-v3',
    'NF-UNSW-NB15-v3',
    'NF-CSE-CIC-IDS2018-v3',
    'BigFlow-NIDS'
]

DATASET_FILES = {
    'NF-ToN-IoT-v3': DATA_DIR / 'NF-ToN-IoT-v3.csv',
    'NF-BoT-IoT-v3': DATA_DIR / 'NF-BoT-IoT-v3.csv',
    'NF-UNSW-NB15-v3': DATA_DIR / 'NF-UNSW-NB15-v3.csv',
    'NF-CSE-CIC-IDS2018-v3': DATA_DIR / 'NF-CSE-CIC-IDS2018-v3.csv',
    'BigFlow-NIDS': DATA_DIR / 'BigFlow-NIDS.csv'
}

# Network identifiers to exclude from model inputs
IDENTIFIER_COLS = ['IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT']

# Timestamps for chronological ordering ONLY (excluded from model input features)
TIMESTAMP_COLS = ['FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS']

LABEL_COL = 'Label'
ATTACK_COL = 'Attack'

# Temporal Subsampling & Windowing Settings
SUBSET_SIZE = 500_000
WINDOW_SIZE = 10         # W = 10 consecutive flows
TRAIN_STRIDE = 5         # S = 5 for training
VAL_STRIDE = 1           # S = 1 for validation
TEST_STRIDE = 1          # S = 1 for test
SPLIT_BOUNDARY_GAP = 10  # 10 flows gap between splits to eliminate boundary overlap

# Train / Val / Test Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training Hyperparameters
BATCH_SIZE = 2048
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 0

# Model-specific parameters
KAN_GRID_SIZE = 5
KAN_SPLINE_ORDER = 2
KAN_WINDOW_HIDDEN = [32, 16]
KAN_STATIC_HIDDEN = [64, 32]

RNN_HIDDEN_DIM = 64
RNN_NUM_LAYERS = 2
RNN_DROPOUT = 0.3

TRANSFORMER_D_MODEL = 48
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_FFN_DIM = 96
TRANSFORMER_DROPOUT = 0.2
