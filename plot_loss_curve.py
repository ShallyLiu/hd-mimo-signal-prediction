"""
Training curve visualization module
Plot training and validation loss curves
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict, Tuple
import os

# Set font style
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_loss_curves(train_losses: List[float],
                    val_losses: List[float],
                    title: str = "Training and Validation Loss",
                    save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 6),
                    log_scale: bool = False) -> None:
    """
    Plot training and validation loss curves with performance annotations
    
    Args:
        train_losses: Training loss list
        val_losses: Validation loss list
        title: Plot title
        save_path: Save path
        figsize: Figure size
        log_scale: Whether to use log scale
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=figsize)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2.5, alpha=0.8)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2.5, alpha=0.8)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.yscale('log')
    
    # Enhanced annotations with final performance
    min_val_idx = np.argmin(val_losses)
    min_val_loss = val_losses[min_val_idx]
    final_val_loss = val_losses[-1]
    
    # Best validation loss annotation
    plt.annotate(f'Best Val Loss: {min_val_loss:.6f}\nEpoch: {min_val_idx+1}',
                xy=(min_val_idx+1, min_val_loss),
                xytext=(min_val_idx+1+len(epochs)*0.15, min_val_loss*1.2 if not log_scale else min_val_loss*2),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7, lw=1.5),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8, edgecolor='red'))
    
    # Final performance annotation
    plt.annotate(f'Final Val Loss: {final_val_loss:.6f}',
                xy=(len(epochs), final_val_loss),
                xytext=(len(epochs)-len(epochs)*0.2, final_val_loss*1.5 if not log_scale else final_val_loss*3),
                arrowprops=dict(arrowstyle='->', color='darkred', alpha=0.7, lw=1.5),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8, edgecolor='darkred'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss curve saved: {save_path}")
    
    plt.show()

def plot_nmse_curves(train_nmse: List[float],
                    val_nmse: List[float],
                    title: str = "Training and Validation NMSE",
                    save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot training and validation NMSE curves with performance annotations
    
    Args:
        train_nmse: Training NMSE list
        val_nmse: Validation NMSE list
        title: Plot title
        save_path: Save path
        figsize: Figure size
    """
    epochs = range(1, len(train_nmse) + 1)
    
    # Convert to dB
    train_nmse_db = [10 * np.log10(nmse + 1e-12) for nmse in train_nmse]
    val_nmse_db = [10 * np.log10(nmse + 1e-12) for nmse in val_nmse]
    
    plt.figure(figsize=figsize)
    
    plt.plot(epochs, train_nmse_db, 'b-', label='Training NMSE', linewidth=2.5, alpha=0.8)
    plt.plot(epochs, val_nmse_db, 'r-', label='Validation NMSE', linewidth=2.5, alpha=0.8)
    
    plt.xlabel('Epoch')
    plt.ylabel('NMSE (dB)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Enhanced annotations
    min_val_idx = np.argmin(val_nmse_db)
    min_val_nmse = val_nmse_db[min_val_idx]
    final_val_nmse = val_nmse_db[-1]
    
    # Best validation NMSE annotation
    plt.annotate(f'Best Val NMSE: {min_val_nmse:.2f} dB\nEpoch: {min_val_idx+1}',
                xy=(min_val_idx+1, min_val_nmse),
                xytext=(min_val_idx+1+len(epochs)*0.15, min_val_nmse+2),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7, lw=1.5),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8, edgecolor='red'))
    
    # Final performance annotation with convergence point
    plt.annotate(f'Final NMSE: {final_val_nmse:.2f} dB\nConverged',
                xy=(len(epochs), final_val_nmse),
                xytext=(len(epochs)-len(epochs)*0.2, final_val_nmse+3),
                arrowprops=dict(arrowstyle='->', color='darkred', alpha=0.7, lw=1.5),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8, edgecolor='darkred'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"NMSE curve saved: {save_path}")
    
    plt.show()

def plot_combined_metrics(train_losses: List[float],
                         val_losses: List[float],
                         train_nmse: List[float],
                         val_nmse: List[float],
                         title: str = "Training Metrics",
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Plot combined metrics with enhanced annotations
    
    Args:
        train_losses: Training loss list
        val_losses: Validation loss list
        train_nmse: Training NMSE list
        val_nmse: Validation NMSE list
        title: Plot title
        save_path: Save path
        figsize: Figure size
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2.5, alpha=0.8)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2.5, alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss annotations
    min_val_loss_idx = np.argmin(val_losses)
    min_val_loss = val_losses[min_val_loss_idx]
    ax1.annotate(f'Best: {min_val_loss:.4f}',
                xy=(min_val_loss_idx+1, min_val_loss),
                xytext=(min_val_loss_idx+1+len(epochs)*0.1, min_val_loss*1.2),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))
    
    # NMSE curves (dB)
    train_nmse_db = [10 * np.log10(nmse + 1e-12) for nmse in train_nmse]
    val_nmse_db = [10 * np.log10(nmse + 1e-12) for nmse in val_nmse]
    
    ax2.plot(epochs, train_nmse_db, 'b-', label='Training NMSE', linewidth=2.5, alpha=0.8)
    ax2.plot(epochs, val_nmse_db, 'r-', label='Validation NMSE', linewidth=2.5, alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('NMSE (dB)')
    ax2.set_title('NMSE Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # NMSE annotations
    min_val_nmse_idx = np.argmin(val_nmse_db)
    min_val_nmse = val_nmse_db[min_val_nmse_idx]
    final_val_nmse = val_nmse_db[-1]
    
    ax2.annotate(f'Best: {min_val_nmse:.1f} dB',
                xy=(min_val_nmse_idx+1, min_val_nmse),
                xytext=(min_val_nmse_idx+1+len(epochs)*0.1, min_val_nmse+1.5),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))
    
    ax2.annotate(f'Final: {final_val_nmse:.1f} dB',
                xy=(len(epochs), final_val_nmse),
                xytext=(len(epochs)-len(epochs)*0.15, final_val_nmse+2),
                arrowprops=dict(arrowstyle='->', color='darkred', alpha=0.7),
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcoral", alpha=0.7))
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined metrics plot saved: {save_path}")
    
    plt.show()

def plot_learning_rate_schedule(learning_rates: List[float],
                               title: str = "Learning Rate Schedule",
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot learning rate schedule curve
    
    Args:
        learning_rates: Learning rate list
        title: Plot title
        save_path: Save path
        figsize: Figure size
    """
    epochs = range(1, len(learning_rates) + 1)
    
    plt.figure(figsize=figsize)
    plt.plot(epochs, learning_rates, 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning rate curve saved: {save_path}")
    
    plt.show()

def plot_ensemble_weight_analysis(weights_range: np.ndarray,
                                mse_values: List[float],
                                nmse_values: List[float],
                                optimal_weight: float,
                                title: str = "Ensemble Weight Analysis",
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 5)) -> None:
    """
    Plot ensemble weight analysis
    
    Args:
        weights_range: Weight range
        mse_values: MSE values list
        nmse_values: NMSE values list
        optimal_weight: Optimal weight
        title: Plot title
        save_path: Save path
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # MSE vs weight
    ax1.plot(weights_range, mse_values, 'b-', linewidth=2)
    ax1.axvline(x=optimal_weight, color='r', linestyle='--', 
               label=f'Optimal α = {optimal_weight:.3f}')
    ax1.set_xlabel('Ensemble Weight (α)')
    ax1.set_ylabel('MSE')
    ax1.set_title('MSE vs Ensemble Weight')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # NMSE vs weight
    nmse_db = [10 * np.log10(nmse + 1e-12) for nmse in nmse_values]
    ax2.plot(weights_range, nmse_db, 'g-', linewidth=2)
    ax2.axvline(x=optimal_weight, color='r', linestyle='--',
               label=f'Optimal α = {optimal_weight:.3f}')
    ax2.set_xlabel('Ensemble Weight (α)')
    ax2.set_ylabel('NMSE (dB)')
    ax2.set_title('NMSE vs Ensemble Weight')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Ensemble weight analysis plot saved: {save_path}")
    
    plt.show()

def plot_model_comparison_bar(model_metrics: Dict[str, Dict[str, float]],
                            metrics_to_plot: List[str] = ['mse', 'nmse', 'correlation'],
                            title: str = "Model Performance Comparison",
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot model performance comparison bar chart
    
    Args:
        model_metrics: Model metrics dictionary
        metrics_to_plot: Metrics to plot list
        title: Plot title
        save_path: Save path
        figsize: Figure size
    """
    models = list(model_metrics.keys())
    n_metrics = len(metrics_to_plot)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightpink']
    
    for i, metric in enumerate(metrics_to_plot):
        values = [model_metrics[model][metric] for model in models]
        
        bars = axes[i].bar(models, values, color=colors[:len(models)])
        axes[i].set_title(f'{metric.upper()}')
        axes[i].set_ylabel('Value')
        axes[i].grid(True, alpha=0.3)
        
        # Add value annotations on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if metric == 'nmse':
                display_value = f'{10 * np.log10(value + 1e-12):.1f} dB'
            else:
                display_value = f'{value:.4f}'
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        display_value, ha='center', va='bottom')
        
        # Rotate x-axis labels
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved: {save_path}")
    
    plt.show()

def load_and_plot_training_history(history_path: str,
                                  model_name: str,
                                  save_dir: str = "results") -> None:
    """
    Load and plot training history
    
    Args:
        history_path: Training history file path
        model_name: Model name
        save_dir: Save directory
    """
    if not os.path.exists(history_path):
        print(f"Training history file not found: {history_path}")
        return
    
    history = np.load(history_path, allow_pickle=True).item()
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot combined metrics
    combined_save_path = os.path.join(save_dir, f"training_curves_{model_name.lower()}.png")
    plot_combined_metrics(
        history['train_losses'],
        history['val_losses'],
        history['train_nmse'],
        history['val_nmse'],
        title=f"{model_name} Training Metrics",
        save_path=combined_save_path
    )

def create_all_training_plots(save_dir: str = "results") -> None:
    """
    Create all training plots
    
    Args:
        save_dir: Save directory
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load training histories
    models = ['rescnn', 'transformer']
    
    for model in models:
        history_path = f"checkpoints/{model}_training_history.npy"
        model_display_name = model.replace('rescnn', 'ResCNN').replace('transformer', 'Transformer')
        load_and_plot_training_history(history_path, model_display_name, save_dir)

if __name__ == "__main__":
    # Test plotting functions
    print("Testing training curve plotting functions...")
    
    # Generate test data with realistic convergence patterns
    epochs = 100
    
    # ResCNN-style convergence
    rescnn_train_losses = [0.8 * np.exp(-i/15) + 0.05 + 0.02*np.random.random() for i in range(epochs)]
    rescnn_val_losses = [0.8 * np.exp(-i/18) + 0.08 + 0.025*np.random.random() for i in range(epochs)]
    rescnn_train_nmse = [0.4 * np.exp(-i/12) + 0.03 + 0.01*np.random.random() for i in range(epochs)]
    rescnn_val_nmse = [0.4 * np.exp(-i/15) + 0.05 + 0.015*np.random.random() for i in range(epochs)]
    
    # Test plots
    plot_combined_metrics(
        rescnn_train_losses, rescnn_val_losses, rescnn_train_nmse, rescnn_val_nmse,
        title="ResCNN Training Metrics"
    )
    
    # Test ensemble weight analysis
    weights = np.linspace(0, 1, 21)
    mse_vals = [0.1 + 0.05 * (w - 0.6)**2 for w in weights]
    nmse_vals = [0.05 + 0.02 * (w - 0.6)**2 for w in weights]
    plot_ensemble_weight_analysis(weights, mse_vals, nmse_vals, 0.6)
    
    print("Training curve plotting test completed!")