"""
Shared Utility Functions for Brain Tumor Classification

This module provides reusable functions for:
- Data loading and transformations
- Training and validation loops
- Evaluation metrics and visualization
- Reproducibility setup
- Model checkpointing

All experiments should use these functions to ensure consistency.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import yaml
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path: str = 'config.yaml') -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set all random seeds for reproducibility.
    
    This sets seeds for:
    - Python random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - cuDNN (deterministic mode)
    
    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but reproducible)
    """
    print(f"Setting random seed to {seed} (deterministic={deterministic})")
    
    # Python
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # cuDNN
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    print("✓ All random seeds set successfully")


def get_transforms(config: dict, split: str = 'train') -> transforms.Compose:
    """
    Get data transforms for specified split.
    
    IMPORTANT: Transform order is critical!
    1. Augmentations (on PIL Image) - only for training
    2. ToTensor (PIL → Tensor)
    3. Normalize (on Tensor)
    
    Args:
        config: Configuration dictionary
        split: 'train', 'val', or 'test'
        
    Returns:
        Composed transforms
    """
    data_config = config['data']
    img_size = data_config['image_size']
    
    if split == 'train':
        # Training: augmentations → ToTensor → normalize
        aug_config = data_config['augmentation']
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=aug_config['random_horizontal_flip']['prob']),
            transforms.RandomRotation(degrees=aug_config['random_rotation']['degrees']),
            transforms.RandomAffine(
                degrees=aug_config['random_affine']['degrees'],
                translate=tuple(aug_config['random_affine']['translate']),
                scale=tuple(aug_config['random_affine']['scale'])
            ),
            transforms.ColorJitter(
                brightness=aug_config['color_jitter']['brightness'],
                contrast=aug_config['color_jitter']['contrast'],
                saturation=aug_config['color_jitter']['saturation'],
                hue=aug_config['color_jitter']['hue']
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=data_config['normalization']['mean'],
                std=data_config['normalization']['std']
            )
        ]
    else:
        # Validation/Test: only resize → ToTensor → normalize
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=data_config['normalization']['mean'],
                std=data_config['normalization']['std']
            )
        ]
    
    return transforms.Compose(transform_list)


def get_dataloaders(config: dict, preprocessed_dir: str) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, val, internal_test, and heldout_test sets.
    
    Args:
        config: Configuration dictionary
        preprocessed_dir: Path to preprocessed_canonical directory
        
    Returns:
        Tuple of (train_loader, val_loader, internal_test_loader, heldout_test_loader)
    """
    train_config = config['training']
    
    # Get transforms
    train_transform = get_transforms(config, 'train')
    eval_transform = get_transforms(config, 'val')
    
    # Create datasets
    train_dataset = datasets.ImageFolder(
        root=os.path.join(preprocessed_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(preprocessed_dir, 'val'),
        transform=eval_transform
    )
    
    internal_test_dataset = datasets.ImageFolder(
        root=os.path.join(preprocessed_dir, 'internal_test'),
        transform=eval_transform
    )
    
    heldout_test_dataset = datasets.ImageFolder(
        root=os.path.join(preprocessed_dir, 'heldout_test'),
        transform=eval_transform
    )
    
    # Worker init function for reproducibility
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=train_config['num_workers'],
        pin_memory=train_config['pin_memory'],
        worker_init_fn=worker_init_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config['num_workers'],
        pin_memory=train_config['pin_memory']
    )
    
    internal_test_loader = DataLoader(
        internal_test_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config['num_workers'],
        pin_memory=train_config['pin_memory']
    )
    
    heldout_test_loader = DataLoader(
        heldout_test_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config['num_workers'],
        pin_memory=train_config['pin_memory']
    )
    
    print(f"✓ DataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Internal Test: {len(internal_test_dataset)} samples, {len(internal_test_loader)} batches")
    print(f"  Heldout Test: {len(heldout_test_dataset)} samples, {len(heldout_test_loader)} batches")
    
    return train_loader, val_loader, internal_test_loader, heldout_test_loader


def validate_checkpoint_fresh(checkpoint_path: str, force_fresh: bool = True):
    """
    Validate that checkpoint doesn't exist (ensuring fresh training).
    
    Args:
        checkpoint_path: Path where checkpoint will be saved
        force_fresh: If True, raise error if checkpoint exists
    """
    if os.path.exists(checkpoint_path):
        if force_fresh:
            raise FileExistsError(
                f"Checkpoint already exists: {checkpoint_path}\n"
                f"To ensure fresh training, delete this checkpoint first or use a different name."
            )
        else:
            print(f"⚠️  WARNING: Checkpoint exists: {checkpoint_path}")
            print(f"    Training will overwrite this checkpoint.")
    else:
        print(f"✓ Checkpoint path is fresh: {checkpoint_path}")


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    max_grad_norm: float = 1.0,
    epoch: int = 0
) -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scaler: GradScaler for mixed precision (optional)
        max_grad_norm: Maximum gradient norm for clipping
        epoch: Current epoch number (for logging)
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training if scaler provided
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate model on validation set.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: List[str]
) -> Dict:
    """
    Comprehensive model evaluation with metrics.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        class_names: List of class names
        
    Returns:
        Dictionary containing all metrics, predictions, and labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Accuracy
    accuracy = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
    
    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = 'Confusion Matrix'
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure (optional)
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc' lists
        save_path: Path to save figure (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to {save_path}")
    
    plt.show()


def plot_roc_curves(
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
):
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        labels: True labels
        probs: Predicted probabilities (n_samples, n_classes)
        class_names: List of class names
        save_path: Path to save figure (optional)
    """
    n_classes = len(class_names)
    
    # Binarize labels
    labels_bin = label_binarize(labels, classes=range(n_classes))
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i], tpr[i],
            color=color,
            lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-Class Classification', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curves saved to {save_path}")
    
    plt.show()
    
    return roc_auc


def save_results(
    results: Dict,
    config: Dict,
    save_dir: str,
    experiment_name: str
):
    """
    Save experiment results to JSON file.
    
    Args:
        results: Results dictionary
        config: Configuration used
        save_dir: Directory to save results
        experiment_name: Name of experiment
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare results for JSON (convert numpy to lists)
    results_json = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'config': config,
        'accuracy': float(results['accuracy']),
        'classification_report': results['classification_report'],
        'confusion_matrix': results['confusion_matrix'].tolist()
    }
    
    # Save
    results_path = os.path.join(save_dir, f'{experiment_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"✓ Results saved to {results_path}")


def print_experiment_summary(
    model_name: str,
    results: Dict,
    class_names: List[str]
):
    """
    Print formatted experiment summary.
    
    Args:
        model_name: Name of the model
        results: Results dictionary
        class_names: List of class names
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT SUMMARY: {model_name}")
    print("="*80)
    
    print(f"\nOverall Accuracy: {results['accuracy']:.2f}%")
    
    print("\nPer-Class Metrics:")
    print("-" * 60)
    report = results['classification_report']
    for class_name in class_names:
        metrics = report[class_name]
        print(f"{class_name:15} | Precision: {metrics['precision']:.3f} | "
              f"Recall: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f}")
    
    print("\nMacro Average:")
    print("-" * 60)
    macro = report['macro avg']
    print(f"Precision: {macro['precision']:.3f} | Recall: {macro['recall']:.3f} | "
          f"F1: {macro['f1-score']:.3f}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print("This is a utility module. Import functions as needed.")
    print("\nAvailable functions:")
    print("  - load_config(): Load configuration from YAML")
    print("  - set_seed(): Set all random seeds for reproducibility")
    print("  - get_transforms(): Get data transforms for train/val/test")
    print("  - get_dataloaders(): Create DataLoaders")
    print("  - validate_checkpoint_fresh(): Ensure checkpoint doesn't exist")
    print("  - train_one_epoch(): Train for one epoch")
    print("  - validate(): Validate model")
    print("  - evaluate_model(): Comprehensive evaluation")
    print("  - plot_confusion_matrix(): Plot confusion matrix")
    print("  - plot_training_curves(): Plot training curves")
    print("  - plot_roc_curves(): Plot ROC curves")
    print("  - save_results(): Save results to JSON")
    print("  - print_experiment_summary(): Print formatted summary")
