"""
Training module for Temporal NetFlow Benchmark v3.
Implements unified training loop for KAN-static, KAN-window, LSTM, GRU, and Transformer.
Uses class-weighted cross-entropy loss and validation checkpointing.
"""
import sys
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = CURRENT_DIR.parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from . import config
from .utils import EarlyStopping, format_time, plot_training_history, count_parameters


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device) -> Dict[str, float]:
    """Train model for a single epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += batch_size
        
    return {
        'loss': total_loss / max(1, total),
        'accuracy': 100.0 * correct / max(1, total)
    }


def validate_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                       device: torch.device) -> Dict[str, float]:
    """Evaluate model on validation loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += batch_size
            
    return {
        'loss': total_loss / max(1, total),
        'accuracy': 100.0 * correct / max(1, total)
    }


def train_temporal_model(model: nn.Module,
                         X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         run_name: str,
                         device: torch.device,
                         class_weights: Optional[torch.Tensor] = None,
                         num_epochs: int = config.NUM_EPOCHS,
                         batch_size: int = config.BATCH_SIZE,
                         learning_rate: float = config.LEARNING_RATE,
                         patience: int = config.EARLY_STOPPING_PATIENCE) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train a temporal model and checkpoint the best weights based on validation loss.
    """
    model = model.to(device)
    param_count = count_parameters(model)
    print(f"\nTraining [{run_name}] | Parameters: {param_count:,} | Device: {device}", flush=True)
    print(f"  Train windows: {len(y_train):,} | Val windows: {len(y_val):,}", flush=True)
    
    # Create DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=(device.type == 'cuda')
    )
    
    # Loss criterion with class weighting
    if class_weights is not None:
        if not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    early_stopping = EarlyStopping(patience=patience)
    
    checkpoint_path = config.CHECKPOINTS_DIR / f"{run_name}_best.pt"
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_res = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_res = validate_one_epoch(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_res['loss'])
        history['train_acc'].append(train_res['accuracy'])
        history['val_loss'].append(val_res['loss'])
        history['val_acc'].append(val_res['accuracy'])
        
        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1:2d}/{num_epochs:2d} | "
              f"Train Loss: {train_res['loss']:.4f} Acc: {train_res['accuracy']:.2f}% | "
              f"Val Loss: {val_res['loss']:.4f} Acc: {val_res['accuracy']:.2f}% | "
              f"LR: {lr_now:.2e}", flush=True)
              
        scheduler.step(val_res['loss'])
        
        # Save best checkpoint
        if val_res['loss'] < best_val_loss:
            best_val_loss = val_res['loss']
            best_val_acc = val_res['accuracy']
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    [*] Checkpoint saved (val_loss: {best_val_loss:.4f}, val_acc: {best_val_acc:.2f}%)", flush=True)
            
        early_stopping(val_res['loss'])
        if early_stopping.early_stop:
            print("    [!] Early stopping triggered.", flush=True)
            break
            
    training_time = time.time() - start_time
    print(f"Finished {run_name} in {format_time(training_time)} | Best Val Loss: {best_val_loss:.4f}", flush=True)
    
    # Reload best weights
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
        print(f"Loaded optimal weights from {checkpoint_path.name}", flush=True)
        
    # Plot training curves
    plot_path = config.FIGURES_DIR / f"{run_name}_history.png"
    plot_training_history(history, save_path=str(plot_path))
    
    meta = {
        'run_name': run_name,
        'history': history,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'training_time_sec': training_time,
        'parameters': param_count
    }
    return model, meta
