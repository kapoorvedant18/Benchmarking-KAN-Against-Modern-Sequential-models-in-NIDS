"""
Data loader and temporal window generator for Temporal NetFlow Benchmark v3.
Implements:
1. Strict chronological sorting by FLOW_START_MILLISECONDS.
2. Contiguous chronological subset of ~500,000 flows (no random row shuffling/sampling).
3. Sliding windows: W=10 consecutive flows in time order (stride 5 for train, stride 1 for val/test).
4. Window label = label of the last flow in the window (flow at index t).
5. 70/15/15 time-ordered splits with a gap of W=10 flows between splits to prevent boundary overlap.
6. StandardScaler fit strictly on train split only.
7. Class-weighted cross-entropy loss computation (no SMOTE).
8. Exclusion of identifiers and timestamps from model inputs (47 NetFlow v3 features retained).
"""
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = CURRENT_DIR.parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from . import config


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw dataframe: drop identifier columns, coerce features to numeric,
    filter out infinite and extreme counter overflow values (> 1e14).
    """
    df = df.copy()
    
    # Drop identifier columns
    cols_to_drop = [c for c in config.IDENTIFIER_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        
    df = df.dropna()
    
    target_cols = [c for c in [config.LABEL_COL, config.ATTACK_COL] if c in df.columns]
    time_cols = [c for c in config.TIMESTAMP_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in target_cols + time_cols]
    
    # Ensure timestamps are numeric
    for t_col in time_cols:
        df[t_col] = pd.to_numeric(df[t_col], errors='coerce')
        
    # Ensure features are numeric
    for f_col in feature_cols:
        df[f_col] = pd.to_numeric(df[f_col], errors='coerce')
        
    # Mask counter overflows > 1e14 and infs
    df[feature_cols] = df[feature_cols].mask(df[feature_cols].abs() > 1e14)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    return df


def load_chronological_subset(dataset_name: str, subset_size: int = config.SUBSET_SIZE) -> pd.DataFrame:
    """
    Load a contiguous chronological block of flows from the dataset.
    Sorts strictly by FLOW_START_MILLISECONDS to preserve genuine temporal ordering.
    """
    csv_file = config.DATASET_FILES.get(dataset_name)
    if csv_file is None or not csv_file.exists():
        raise FileNotFoundError(f"Dataset file for {dataset_name} not found at {csv_file}")
        
    print(f"Reading contiguous chronological flows from {csv_file.name}...", flush=True)
    # Read subset_size + buffer to account for NaN cleaning
    buffer_size = int(subset_size * 1.05)
    df = pd.read_csv(csv_file, nrows=buffer_size)
    print(f"  Read {len(df):,} raw flows. Cleaning...", flush=True)
    
    df = clean_dataframe(df)
    
    # Ensure chronological order by FLOW_START_MILLISECONDS
    if 'FLOW_START_MILLISECONDS' in df.columns:
        if not df['FLOW_START_MILLISECONDS'].is_monotonic_increasing:
            print("  Sorting flows strictly by FLOW_START_MILLISECONDS...", flush=True)
            df = df.sort_values(by='FLOW_START_MILLISECONDS').reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
            
    # Take exactly subset_size contiguous flows
    if len(df) > subset_size:
        df = df.iloc[:subset_size].copy()
        
    print(f"  Final contiguous chronological slice: {len(df):,} flows.", flush=True)
    return df


def generate_sliding_windows(X: np.ndarray, y: np.ndarray, window_size: int = 10, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sliding windows of shape (num_windows, window_size, num_features).
    Window label = label of the last flow in the window (flow at index t).
    """
    N, D = X.shape
    if N < window_size:
        raise ValueError(f"Array length {N} is less than window size {window_size}")
        
    # sliding_window_view on axis 0 gives (N - window_size + 1, D, window_size)
    # swapaxes(1, 2) produces (N - window_size + 1, window_size, D)
    windows = sliding_window_view(X, window_shape=window_size, axis=0).swapaxes(1, 2)
    
    # Apply stride
    windows_strided = windows[::stride]
    
    # Label is the last flow of each window: index window_size - 1 + i * stride
    labels_strided = y[window_size - 1 :: stride][:len(windows_strided)]
    
    return np.ascontiguousarray(windows_strided), np.ascontiguousarray(labels_strided)


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for cross-entropy loss.
    Handles unrepresented classes safely with weight 1.0.
    """
    classes, counts = np.unique(y_train, return_counts=True)
    weight_dict = {cls: count for cls, count in zip(classes, counts)}
    
    weights = np.ones(num_classes, dtype=np.float32)
    total_samples = len(y_train)
    
    for cls in range(num_classes):
        if cls in weight_dict and weight_dict[cls] > 0:
            # Standard balanced weighting: N / (C * N_c)
            weights[cls] = total_samples / (len(classes) * weight_dict[cls])
        else:
            weights[cls] = 1.0
            
    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def prepare_temporal_benchmark_data(df: pd.DataFrame, mode: str = 'binary') -> Dict[str, Any]:
    """
    Full temporal data preparation pipeline:
    1. Label extraction (binary vs multiclass).
    2. Strict 70% train / 15% val / 15% test chronological partition.
    3. Gap of W=10 flows between splits to prevent sequence boundary leakage.
    4. StandardScaler fit strictly on train split only.
    5. Sliding windows: W=10 (stride 5 for train, stride 1 for val/test).
    6. Class weights computation for CrossEntropyLoss.
    """
    print(f"\n--- Preparing Temporal Data ({mode.upper()}) ---", flush=True)
    
    # Extract labels and classes
    if mode == 'binary':
        if config.LABEL_COL not in df.columns:
            raise ValueError(f"Missing {config.LABEL_COL} in dataset.")
        y_all = df[config.LABEL_COL].astype(int).values
        class_names = ['Benign', 'Attack']
    elif mode == 'multiclass':
        if config.ATTACK_COL not in df.columns:
            raise ValueError(f"Missing {config.ATTACK_COL} in dataset.")
        le = LabelEncoder()
        y_all = le.fit_transform(df[config.ATTACK_COL].astype(str))
        class_names = list(le.classes_)
    else:
        raise ValueError(f"Unknown mode: {mode}")
        
    num_classes = len(class_names)
    
    # Separate input feature columns (drop identifiers, timestamps, labels)
    exclude_cols = set(config.IDENTIFIER_COLS + config.TIMESTAMP_COLS + [config.LABEL_COL, config.ATTACK_COL])
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    num_features = len(feature_cols)
    
    print(f"  Input Features ({num_features}): {feature_cols[:5]} ... {feature_cols[-3:]}")
    print(f"  Classes ({num_classes}): {class_names}")
    
    # Compute chronological split indices
    N = len(df)
    n_train = int(config.TRAIN_RATIO * N)
    n_val = int(config.VAL_RATIO * N)
    gap = config.SPLIT_BOUNDARY_GAP
    
    # Boundaries:
    # Train: 0 -> n_train
    # Gap 1: n_train -> n_train + gap (dropped)
    # Val: n_train + gap -> n_train + gap + n_val
    # Gap 2: n_train + gap + n_val -> n_train + gap + n_val + gap (dropped)
    # Test: n_train + gap + n_val + gap -> N
    train_idx_end = n_train
    val_idx_start = n_train + gap
    val_idx_end = val_idx_start + n_val
    test_idx_start = val_idx_end + gap
    test_idx_end = N
    
    X_raw = df[feature_cols].to_numpy(dtype=np.float32)
    
    X_train_raw = X_raw[:train_idx_end]
    y_train_raw = y_all[:train_idx_end]
    
    X_val_raw = X_raw[val_idx_start:val_idx_end]
    y_val_raw = y_all[val_idx_start:val_idx_end]
    
    X_test_raw = X_raw[test_idx_start:test_idx_end]
    y_test_raw = y_all[test_idx_start:test_idx_end]
    
    print(f"  Raw Chronological Splits:")
    print(f"    Train: {len(X_train_raw):,} flows [0 : {train_idx_end}]")
    print(f"    Gap 1: {gap} flows dropped [{train_idx_end} : {val_idx_start}]")
    print(f"    Val:   {len(X_val_raw):,} flows [{val_idx_start} : {val_idx_end}]")
    print(f"    Gap 2: {gap} flows dropped [{val_idx_end} : {test_idx_start}]")
    print(f"    Test:  {len(X_test_raw):,} flows [{test_idx_start} : {test_idx_end}]")
    
    # Fit StandardScaler strictly on training set
    scaler = StandardScaler()
    X_train_scaled = np.nan_to_num(scaler.fit_transform(X_train_raw), nan=0.0, posinf=1.0, neginf=-1.0)
    X_val_scaled = np.nan_to_num(scaler.transform(X_val_raw), nan=0.0, posinf=1.0, neginf=-1.0)
    X_test_scaled = np.nan_to_num(scaler.transform(X_test_raw), nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Generate sliding windows
    print(f"  Generating sliding windows (W={config.WINDOW_SIZE})...", flush=True)
    X_train_w, y_train_w = generate_sliding_windows(
        X_train_scaled, y_train_raw, window_size=config.WINDOW_SIZE, stride=config.TRAIN_STRIDE
    )
    X_val_w, y_val_w = generate_sliding_windows(
        X_val_scaled, y_val_raw, window_size=config.WINDOW_SIZE, stride=config.VAL_STRIDE
    )
    X_test_w, y_test_w = generate_sliding_windows(
        X_test_scaled, y_test_raw, window_size=config.WINDOW_SIZE, stride=config.TEST_STRIDE
    )
    
    print(f"    Train Windows: {X_train_w.shape} (stride {config.TRAIN_STRIDE})")
    print(f"    Val Windows:   {X_val_w.shape} (stride {config.VAL_STRIDE})")
    print(f"    Test Windows:  {X_test_w.shape} (stride {config.TEST_STRIDE})")
    
    # Compute class weights for training
    class_weights = compute_class_weights(y_train_w, num_classes)
    
    return {
        'X_train': X_train_w,
        'y_train': y_train_w,
        'X_val': X_val_w,
        'y_val': y_val_w,
        'X_test': X_test_w,
        'y_test': y_test_w,
        'class_weights': class_weights,
        'class_names': class_names,
        'feature_names': feature_cols,
        'num_features': num_features,
        'num_classes': num_classes,
        'scaler': scaler,
        'mode': mode
    }
