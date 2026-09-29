"""
Utility functions for reproducibility, metrics calculation, and plotting.
Strictly self-contained within temporal_benchmark_v3.
"""
import random
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from pathlib import Path


def set_seed(seed: int = 42):
    """
    Ensure complete reproducibility across all libraries and hardware backends.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return active torch device (GPU if available)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_history(history: dict, save_path: str):
    """Plot train/val loss and accuracy curves."""
    has_acc = 'train_acc' in history and 'val_acc' in history
    ncols = 2 if has_acc else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    
    if ncols == 1:
        axes = [axes]
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'], label='Validation')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if has_acc:
        axes[1].plot(epochs, history['train_acc'], label='Train')
        axes[1].plot(epochs, history['val_acc'], label='Validation')
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: str, title: str = 'Confusion Matrix'):
    """Plot confusion matrix heatmap."""
    if isinstance(y_true, np.ndarray) and y_true.ndim == 2:
        cm = y_true
    else:
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    plt.figure(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.8)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_score, class_names=None, save_path: str = None, title: str = 'ROC Curve'):
    """Plot ROC curve for binary or multiclass classification."""
    plt.figure(figsize=(10, 8))
    
    if class_names is None or len(class_names) <= 2:
        if isinstance(y_score, np.ndarray) and y_score.ndim > 1:
            y_score = y_score[:, 1] if y_score.shape[1] == 2 else y_score[:, 0]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    else:
        n_classes = len(class_names)
        present_classes = np.unique(y_true)
        if len(present_classes) > 1:
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            for i in range(n_classes):
                if i in present_classes and (y_true_bin[:, i].sum() > 0):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.4f})')
            
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


class EarlyStopping:
    """Early stops training when validation loss stops improving."""
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'  [EarlyStopping] Counter: {self.counter}/{self.patience}', flush=True)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    return f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}"
