"""
Evaluation module for Temporal NetFlow Benchmark v3.
Computes metrics on unseen future test windows, generates confusion matrices,
ROC curves, per-class breakdown CSVs, and updates master comparison tables incrementally.
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_fscore_support
)

CURRENT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = CURRENT_DIR.parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from . import config
from .utils import plot_confusion_matrix, plot_roc_curve


def get_model_predictions(model: torch.nn.Module, X: np.ndarray, y: np.ndarray,
                          device: torch.device, batch_size: int = config.BATCH_SIZE) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions and class probabilities for temporal sequence windows.
    """
    model.eval()
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS)
    
    y_true_list = []
    y_pred_list = []
    y_prob_list = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            y_true_list.append(targets.cpu().numpy())
            y_pred_list.append(preds)
            y_prob_list.append(probs)
            
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    y_prob = np.concatenate(y_prob_list)
    return y_true, y_pred, y_prob


def update_master_table(metrics: Dict[str, Any], mode: str) -> pd.DataFrame:
    """
    Safely append or update run metrics in the master comparison CSV table.
    """
    table_file = config.TABLES_DIR / f"model_comparison_{mode}.csv"
    
    if table_file.exists():
        try:
            df = pd.read_csv(table_file)
            mask = (df['dataset'] == metrics['dataset']) & (df['model'] == metrics['model'])
            df = df[~mask]
            df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
        except Exception:
            df = pd.DataFrame([metrics])
    else:
        df = pd.DataFrame([metrics])
        
    sort_cols = [c for c in ['dataset', 'model'] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols)
        
    df.to_csv(table_file, index=False)
    print(f"  -> Master table updated: {table_file.name} ({len(df)} runs recorded)", flush=True)
    return df


def evaluate_temporal_run(model: torch.nn.Module,
                          X_test: np.ndarray, y_test: np.ndarray,
                          class_names: List[str],
                          dataset_name: str,
                          model_name: str,
                          mode: str,
                          device: torch.device,
                          training_time_sec: float = 0.0,
                          parameters: int = 0) -> Dict[str, Any]:
    """
    Evaluate a single model on test set windows.
    Generates plots, per-class table (if multiclass), and updates master CSV.
    """
    run_id = f"{dataset_name}_{model_name}_{mode}"
    print(f"\nEvaluating [{run_id}] on {len(y_test):,} unseen test windows...", flush=True)
    
    y_true, y_pred, y_prob = get_model_predictions(model, X_test, y_test, device)
    
    metrics = {
        'dataset': dataset_name,
        'model': model_name,
        'mode': mode,
        'parameters': parameters,
        'training_time_sec': round(training_time_sec, 2),
        'accuracy': float(accuracy_score(y_true, y_pred)),
    }
    
    present_classes = np.unique(y_true)
    num_classes = len(class_names)
    
    if mode == 'binary':
        metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
        
        try:
            if len(present_classes) > 1 and y_prob.shape[1] >= 2:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                metrics['roc_auc'] = float('nan')
        except Exception:
            metrics['roc_auc'] = float('nan')
            
    else:  # multiclass
        metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        
        metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        metrics['precision'] = metrics['precision_macro']
        metrics['recall'] = metrics['recall_macro']
        metrics['f1'] = metrics['f1_macro']
        
        try:
            if len(present_classes) > 1 and y_prob.shape[1] == num_classes:
                # OVR AUC over classes present in y_true
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro', labels=list(range(num_classes))))
            else:
                metrics['roc_auc'] = float('nan')
        except Exception:
            metrics['roc_auc'] = float('nan')
            
        # Per-class breakdown CSV
        prec_arr, rec_arr, f1_arr, supp_arr = precision_recall_fscore_support(
            y_true, y_pred, labels=range(num_classes), zero_division=0
        )
        per_class_df = pd.DataFrame({
            'class_index': range(num_classes),
            'class_name': class_names,
            'precision': prec_arr,
            'recall': rec_arr,
            'f1_score': f1_arr,
            'support': supp_arr
        })
        per_class_path = config.TABLES_DIR / f"{dataset_name}_{model_name}_multiclass_per_class.csv"
        per_class_df.to_csv(per_class_path, index=False)
        print(f"  -> Saved per-class breakdown: {per_class_path.name}", flush=True)
        
    # Plot Confusion Matrix
    cm_path = config.FIGURES_DIR / f"{run_id}_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, str(cm_path),
                          title=f"{dataset_name} | {model_name} ({mode.capitalize()})")
    
    # Plot ROC Curve
    roc_path = config.FIGURES_DIR / f"{run_id}_roc_curve.png"
    try:
        plot_roc_curve(y_true, y_prob, class_names=class_names, save_path=str(roc_path),
                       title=f"{dataset_name} | {model_name} ROC Curve")
    except Exception as e:
        print(f"  Could not plot ROC curve: {e}", flush=True)
        
    # Update master CSV table
    update_master_table(metrics, mode=mode)
    
    print(f"  [Metrics] Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | "
          f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | "
          f"AUC: {metrics.get('roc_auc', float('nan')):.4f}", flush=True)
          
    return metrics
