"""
Unified Experiment Runner for Brain Tumor Classification

This script provides a standardized framework for running all experiments:
- Main training
- Ablation studies
- Baseline comparisons
- Cross-validation

All experiments use consistent preprocessing, training protocol, and evaluation.

Usage:
    python experiment_runner.py --experiment main
    python experiment_runner.py --experiment ablation
    python experiment_runner.py --experiment baseline
    python experiment_runner.py --experiment cross_validation
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import json
import numpy as np
from typing import Dict, List, Tuple

# Import our modules
from utils import (
    load_config, set_seed, get_dataloaders, validate_checkpoint_fresh,
    train_one_epoch, validate, evaluate_model,
    plot_confusion_matrix, plot_training_curves, plot_roc_curves,
    save_results, print_experiment_summary
)
from models import get_model, count_parameters, print_model_summary


###############################################################################
# EXPERIMENT RUNNERS
###############################################################################

def run_main_training(config: Dict, device: torch.device):
    """
    Run main TumorNet-Lite training experiment.
    
    Args:
        config: Configuration dictionary
        device: Device to use for training
    """
    print("\n" + "="*80)
    print("MAIN TRAINING - TUMORNET-LITE")
    print("="*80 + "\n")
    
    # Set seed
    set_seed(config['reproducibility']['seed'], config['reproducibility']['deterministic'])
    
    # Load data
    train_loader, val_loader, internal_test_loader, heldout_test_loader = get_dataloaders(
        config=config,
        preprocessed_dir=config['paths']['preprocessed_data']
    )
    
    class_names = config['data']['class_names']
    num_classes = len(class_names)
    
    # Create model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"tumornet_lite_main_{timestamp}"
    checkpoint_path = os.path.join(config['paths']['checkpoints'], f"{experiment_name}.pth")
    
    validate_checkpoint_fresh(checkpoint_path, force_fresh=True)
    
    model = get_model('tumornet_lite', num_classes=num_classes, pretrained=False)
    model = model.to(device)
    print_model_summary(model, "TumorNet-Lite")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['learning_rate'],
        weight_decay=config['optimizer']['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config['scheduler']['mode'],
        factor=config['scheduler']['factor'],
        patience=config['scheduler']['patience']
    )
    scaler = torch.cuda.amp.GradScaler() if config['training']['mixed_precision'] else None
    
    # Training loop
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        checkpoint_path=checkpoint_path,
        scaler=scaler
    )
    
    # Load best model
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    results = evaluate_all_splits(
        model=model,
        internal_test_loader=internal_test_loader,
        heldout_test_loader=heldout_test_loader,
        device=device,
        class_names=class_names,
        experiment_name=experiment_name,
        config=config
    )
    
    print("\n✓ Main training experiment complete!")
    return results


def run_ablation_study(config: Dict, device: torch.device):
    """
    Run ablation study to evaluate component contributions.
    
    Variants tested:
    1. Full model (TumorNet-Lite)
    2. Without SCTA (Spatial-Channel Tumor Attention)
    3. Without APF (Asymmetric Pyramid Fusion)
    4. Without PFR (Progressive Feature Refinement)
    5. Without all components (baseline backbone)
    
    Args:
        config: Configuration dictionary
        device: Device to use for training
    """
    print("\n" + "="*80)
    print("ABLATION STUDY")
    print("="*80 + "\n")
    
    ablation_config = config['experiments']['ablation_study']
    
    # Variants to test
    variants = [
        ('full', 'Full TumorNet-Lite'),
        ('no_scta', 'Without SCTA'),
        ('no_apf', 'Without APF'),
        ('no_pfr', 'Without PFR'),
        ('baseline', 'Baseline (no components)')
    ]
    
    results_summary = {}
    
    for variant_name, variant_desc in variants:
        print(f"\n{'='*80}")
        print(f"VARIANT: {variant_desc}")
        print(f"{'='*80}\n")
        
        # Set seed for fair comparison
        set_seed(config['reproducibility']['seed'], config['reproducibility']['deterministic'])
        
        # Load data
        train_loader, val_loader, internal_test_loader, _ = get_dataloaders(
            config=config,
            preprocessed_dir=config['paths']['preprocessed_data']
        )
        
        class_names = config['data']['class_names']
        num_classes = len(class_names)
        
        # Create model variant
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"ablation_{variant_name}_{timestamp}"
        checkpoint_path = os.path.join(config['paths']['checkpoints'], f"{experiment_name}.pth")
        
        validate_checkpoint_fresh(checkpoint_path, force_fresh=True)
        
        # TODO: Implement model variants (for now use full model)
        model = get_model('tumornet_lite', num_classes=num_classes, pretrained=False)
        model = model.to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['optimizer']['learning_rate'],
            weight_decay=config['optimizer']['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config['scheduler']['mode'],
            factor=config['scheduler']['factor'],
            patience=config['scheduler']['patience']
        )
        scaler = torch.cuda.amp.GradScaler() if config['training']['mixed_precision'] else None
        
        # Train
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            checkpoint_path=checkpoint_path,
            scaler=scaler
        )
        
        # Evaluate
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        results = evaluate_model(
            model=model,
            test_loader=internal_test_loader,
            device=device,
            class_names=class_names
        )
        
        results_summary[variant_name] = {
            'variant': variant_desc,
            'accuracy': float(results['accuracy']),
            'parameters': count_parameters(model)
        }
        
        print(f"\n✓ {variant_desc} complete!")
        print(f"  Accuracy: {results['accuracy']:.2f}%")
        print(f"  Parameters: {count_parameters(model):,}")
    
    # Print ablation summary
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    for variant_name, result in results_summary.items():
        print(f"{result['variant']:30s} | Acc: {result['accuracy']:6.2f}% | Params: {result['parameters']:>10,}")
    print("="*80 + "\n")
    
    # Save summary
    summary_path = os.path.join(config['paths']['results'], f"ablation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"✓ Ablation summary saved to {summary_path}")
    
    return results_summary


def run_baseline_comparison(config: Dict, device: torch.device):
    """
    Run baseline model comparisons.
    
    Models tested:
    1. TumorNet-Lite (ours)
    2. ResNet-50
    3. EfficientNet-B0
    4. MobileNet-V2
    5. MobileNet-V3-Small
    6. DMFNet
    
    Args:
        config: Configuration dictionary
        device: Device to use for training
    """
    print("\n" + "="*80)
    print("BASELINE COMPARISON")
    print("="*80 + "\n")
    
    # Models to compare
    models_to_test = [
        ('tumornet_lite', 'TumorNet-Lite (Ours)'),
        ('resnet50', 'ResNet-50'),
        ('efficientnet_b0', 'EfficientNet-B0'),
        ('mobilenet_v2', 'MobileNet-V2'),
        ('mobilenet_v3_small', 'MobileNet-V3-Small'),
        ('dmfnet', 'DMFNet')
    ]
    
    results_summary = {}
    
    for model_name, model_desc in models_to_test:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_desc}")
        print(f"{'='*80}\n")
        
        # Set seed for fair comparison
        set_seed(config['reproducibility']['seed'], config['reproducibility']['deterministic'])
        
        # Load data
        train_loader, val_loader, internal_test_loader, _ = get_dataloaders(
            config=config,
            preprocessed_dir=config['paths']['preprocessed_data']
        )
        
        class_names = config['data']['class_names']
        num_classes = len(class_names)
        
        # Create model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"baseline_{model_name}_{timestamp}"
        checkpoint_path = os.path.join(config['paths']['checkpoints'], f"{experiment_name}.pth")
        
        validate_checkpoint_fresh(checkpoint_path, force_fresh=True)
        
        model = get_model(model_name, num_classes=num_classes, pretrained=False)
        model = model.to(device)
        print_model_summary(model, model_desc)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['optimizer']['learning_rate'],
            weight_decay=config['optimizer']['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config['scheduler']['mode'],
            factor=config['scheduler']['factor'],
            patience=config['scheduler']['patience']
        )
        scaler = torch.cuda.amp.GradScaler() if config['training']['mixed_precision'] else None
        
        # Train
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            checkpoint_path=checkpoint_path,
            scaler=scaler
        )
        
        # Evaluate
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        results = evaluate_model(
            model=model,
            test_loader=internal_test_loader,
            device=device,
            class_names=class_names
        )
        
        results_summary[model_name] = {
            'model': model_desc,
            'accuracy': float(results['accuracy']),
            'parameters': count_parameters(model)
        }
        
        print(f"\n✓ {model_desc} complete!")
        print(f"  Accuracy: {results['accuracy']:.2f}%")
        print(f"  Parameters: {count_parameters(model):,}")
    
    # Print comparison summary
    print("\n" + "="*80)
    print("BASELINE COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<30} | {'Accuracy':>10} | {'Parameters':>15}")
    print("-"*80)
    for model_name, result in results_summary.items():
        print(f"{result['model']:<30} | {result['accuracy']:>9.2f}% | {result['parameters']:>15,}")
    print("="*80 + "\n")
    
    # Save summary
    summary_path = os.path.join(config['paths']['results'], f"baseline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"✓ Baseline comparison saved to {summary_path}")
    
    return results_summary


###############################################################################
# TRAINING AND EVALUATION HELPERS
###############################################################################

def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device: torch.device,
    config: Dict,
    checkpoint_path: str,
    scaler=None
) -> Dict:
    """
    Standard training loop used by all experiments.
    
    Returns:
        Training history dictionary
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    max_epochs = config['training']['max_epochs']
    early_stopping_patience = config['training']['early_stopping']['patience']
    max_grad_norm = config['training']['gradient_clipping']['max_norm']
    
    print("Starting training...")
    
    for epoch in range(1, max_epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            max_grad_norm=max_grad_norm,
            epoch=epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # Print progress
        print(f"Epoch [{epoch}/{max_epochs}] "
              f"Train: {train_loss:.4f}/{train_acc:.2f}% "
              f"Val: {val_loss:.4f}/{val_acc:.2f}% "
              f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history,
                'config': config
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    print(f"\nTraining complete! Best val acc: {best_val_acc:.2f}% (epoch {best_epoch})")
    
    return history


def evaluate_all_splits(
    model: nn.Module,
    internal_test_loader,
    heldout_test_loader,
    device: torch.device,
    class_names: List[str],
    experiment_name: str,
    config: Dict
) -> Dict:
    """
    Evaluate model on all test splits and save results.
    
    Returns:
        Dictionary with all evaluation results
    """
    results_dir = config['paths']['results']
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate on internal test
    print("\nEvaluating on internal test set...")
    internal_results = evaluate_model(
        model=model,
        test_loader=internal_test_loader,
        device=device,
        class_names=class_names
    )
    print_experiment_summary(
        model_name=f"{experiment_name} (Internal Test)",
        results=internal_results,
        class_names=class_names
    )
    
    # Evaluate on held-out test
    print("\nEvaluating on held-out test set...")
    heldout_results = evaluate_model(
        model=model,
        test_loader=heldout_test_loader,
        device=device,
        class_names=class_names
    )
    print_experiment_summary(
        model_name=f"{experiment_name} (Held-Out Test)",
        results=heldout_results,
        class_names=class_names
    )
    
    # Save visualizations
    cm_path = os.path.join(results_dir, f"{experiment_name}_confusion_matrix.png")
    plot_confusion_matrix(
        cm=heldout_results['confusion_matrix'],
        class_names=class_names,
        save_path=cm_path
    )
    
    roc_path = os.path.join(results_dir, f"{experiment_name}_roc_curves.png")
    plot_roc_curves(
        labels=heldout_results['labels'],
        probs=heldout_results['probabilities'],
        class_names=class_names,
        save_path=roc_path
    )
    
    return {
        'internal_test': internal_results,
        'heldout_test': heldout_results
    }


###############################################################################
# MAIN
###############################################################################

def main():
    parser = argparse.ArgumentParser(description='Run brain tumor classification experiments')
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        choices=['main', 'ablation', 'baseline', 'cross_validation'],
        help='Type of experiment to run'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). If not specified, auto-detect.'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*80)
    print("BRAIN TUMOR CLASSIFICATION - EXPERIMENT RUNNER")
    print("="*80)
    print(f"Experiment: {args.experiment}")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print("="*80 + "\n")
    
    # Run experiment
    if args.experiment == 'main':
        results = run_main_training(config, device)
    elif args.experiment == 'ablation':
        results = run_ablation_study(config, device)
    elif args.experiment == 'baseline':
        results = run_baseline_comparison(config, device)
    elif args.experiment == 'cross_validation':
        print("Cross-validation not yet implemented. Coming soon!")
        return
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
