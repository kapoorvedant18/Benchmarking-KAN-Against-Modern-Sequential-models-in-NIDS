"""
Master Orchestrator for Temporal NetFlow Benchmark v3.
Runs 5 models across 5 NetFlow v3 datasets in Binary and Multiclass settings
under real temporal windowing (W=10 flows, chronological order).
Incremental progress saving, resume-safe, and self-contained within temporal_benchmark_v3/.
"""
import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

BENCHMARK_ROOT = Path(__file__).resolve().parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))
# Ensure current working directory does not shadow temporal_benchmark_v3/src
sys.path = [p for p in sys.path if Path(p).resolve() != Path('.').resolve()]

from src import config
from src.utils import set_seed, get_device, count_parameters, format_time
from src.data_loader import load_chronological_subset, prepare_temporal_benchmark_data
from src.train import train_temporal_model
from src.evaluate import evaluate_temporal_run
from src.models import (
    KANStaticClassifier, KANWindowClassifier,
    LSTMClassifier, GRUClassifier, TransformerClassifier
)

MODELS = ['KAN-static', 'KAN-window', 'LSTM', 'GRU', 'Transformer']
MODES = ['binary', 'multiclass']


def is_run_completed(dataset_name: str, model_name: str, mode: str) -> bool:
    """Check if a benchmark run has already been recorded in the master table."""
    table_path = config.TABLES_DIR / f"model_comparison_{mode}.csv"
    if not table_path.exists():
        return False
    try:
        df = pd.read_csv(table_path)
        if 'dataset' in df.columns and 'model' in df.columns:
            match = df[(df['dataset'] == dataset_name) & (df['model'] == model_name)]
            return len(match) > 0
    except Exception:
        return False
    return False


def build_model(model_name: str, input_dim: int, num_classes: int) -> torch.nn.Module:
    """Instantiate the requested model architecture with strict parameter parity."""
    if model_name == 'KAN-static':
        return KANStaticClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=config.KAN_STATIC_HIDDEN,
            grid_size=config.KAN_GRID_SIZE,
            spline_order=config.KAN_SPLINE_ORDER
        )
    elif model_name == 'KAN-window':
        return KANWindowClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            window_size=config.WINDOW_SIZE,
            hidden_dims=config.KAN_WINDOW_HIDDEN,
            grid_size=config.KAN_GRID_SIZE,
            spline_order=config.KAN_SPLINE_ORDER
        )
    elif model_name == 'LSTM':
        return LSTMClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=config.RNN_HIDDEN_DIM,
            num_layers=config.RNN_NUM_LAYERS,
            dropout=config.RNN_DROPOUT
        )
    elif model_name == 'GRU':
        return GRUClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=config.RNN_HIDDEN_DIM,
            num_layers=config.RNN_NUM_LAYERS,
            dropout=config.RNN_DROPOUT
        )
    elif model_name == 'Transformer':
        return TransformerClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=config.TRANSFORMER_D_MODEL,
            nhead=config.TRANSFORMER_NHEAD,
            num_layers=config.TRANSFORMER_NUM_LAYERS,
            dim_feedforward=config.TRANSFORMER_FFN_DIM,
            dropout=config.TRANSFORMER_DROPOUT
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_benchmark(target_datasets=None, target_models=None, target_modes=None, force=False):
    """Execute temporal benchmark runs."""
    set_seed(config.RANDOM_SEED)
    device = get_device()
    print("=" * 80)
    print(f"TEMPORAL NETFLOW BENCHMARK v3 (REAL WINDOWS: W={config.WINDOW_SIZE}, STRIDE {config.TRAIN_STRIDE})")
    print(f"Hardware Device: {device} | Random Seed: {config.RANDOM_SEED}")
    print("=" * 80)

    datasets = target_datasets or config.DATASETS
    models = target_models or MODELS
    modes = target_modes or MODES

    total_possible_runs = len(datasets) * len(models) * len(modes)
    completed_runs = 0

    overall_start = time.time()

    for d_idx, dataset_name in enumerate(datasets, 1):
        print("\n" + "#" * 80)
        print(f"[{d_idx}/{len(datasets)}] DATASET: {dataset_name}")
        print("#" * 80)

        # Check if BigFlow-NIDS needs downloading
        if dataset_name == 'BigFlow-NIDS':
            csv_path = config.DATASET_FILES['BigFlow-NIDS']
            if not csv_path.exists():
                print("BigFlow-NIDS CSV not found on disk. Attempting download from S3...", flush=True)
                try:
                    from src.download_bigflow import download_bigflow
                    download_bigflow()
                except Exception as e:
                    print(f"WARNING: Could not download BigFlow-NIDS: {e}. Skipping dataset.", flush=True)
                    continue

        try:
            raw_df = load_chronological_subset(dataset_name, subset_size=config.SUBSET_SIZE)
        except Exception as e:
            print(f"ERROR: Failed to load dataset {dataset_name}: {e}. Skipping.", flush=True)
            continue

        for mode in modes:
            print("\n" + "=" * 60)
            print(f"Task Mode: {mode.upper()} on {dataset_name}")
            print("=" * 60)

            # Check if all models for this dataset & mode are already done
            all_done = all(is_run_completed(dataset_name, m, mode) and not force for m in models)
            if all_done:
                print(f"All {len(models)} models for {dataset_name} ({mode}) already completed. Skipping mode.")
                completed_runs += len(models)
                continue

            # Prepare data splits and sliding windows
            data_dict = prepare_temporal_benchmark_data(raw_df, mode=mode)
            X_train = data_dict['X_train']
            y_train = data_dict['y_train']
            X_val = data_dict['X_val']
            y_val = data_dict['y_val']
            X_test = data_dict['X_test']
            y_test = data_dict['y_test']
            class_weights = data_dict['class_weights']
            class_names = data_dict['class_names']
            num_features = data_dict['num_features']
            num_classes = data_dict['num_classes']

            for m_idx, model_name in enumerate(models, 1):
                run_tag = f"[{dataset_name} | {model_name} | {mode}]"
                print("\n" + "-" * 50)
                print(f"({m_idx}/{len(models)}) Starting {run_tag}")
                print("-" * 50)

                if is_run_completed(dataset_name, model_name, mode) and not force:
                    print(f"{run_tag} already completed. Skipping to next.")
                    completed_runs += 1
                    continue

                # Reset seed before each model run for exact reproducibility
                set_seed(config.RANDOM_SEED)

                model = build_model(model_name, input_dim=num_features, num_classes=num_classes)
                run_name = f"{dataset_name}_{model_name}_{mode}"

                model, train_meta = train_temporal_model(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    run_name=run_name,
                    device=device,
                    class_weights=class_weights,
                    num_epochs=config.NUM_EPOCHS,
                    batch_size=config.BATCH_SIZE,
                    learning_rate=config.LEARNING_RATE,
                    patience=config.EARLY_STOPPING_PATIENCE
                )

                metrics = evaluate_temporal_run(
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    class_names=class_names,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    mode=mode,
                    device=device,
                    training_time_sec=train_meta['training_time_sec'],
                    parameters=train_meta['parameters']
                )

                completed_runs += 1
                elapsed_total = time.time() - overall_start
                print(f"\n>>> PROGRESS: {completed_runs}/{total_possible_runs} runs completed. Total time: {format_time(elapsed_total)} <<<", flush=True)

    print("\n" + "=" * 80)
    print("ALL REQUESTED BENCHMARK RUNS COMPLETED!")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Temporal NetFlow Benchmark v3 Runner")
    parser.add_argument('--dataset', type=str, default='all', choices=config.DATASETS + ['all'])
    parser.add_argument('--model', type=str, default='all', choices=MODELS + ['all'])
    parser.add_argument('--mode', type=str, default='all', choices=MODES + ['all'])
    parser.add_argument('--force', action='store_true', help="Force re-running completed runs")
    args = parser.parse_args()

    datasets = None if args.dataset == 'all' else [args.dataset]
    models = None if args.model == 'all' else [args.model]
    modes = None if args.mode == 'all' else [args.mode]

    run_benchmark(target_datasets=datasets, target_models=models, target_modes=modes, force=args.force)
