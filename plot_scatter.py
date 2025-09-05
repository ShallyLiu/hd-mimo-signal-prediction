"""
Scatter plot visualization module
Plot true vs predicted value scatter plots
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, Tuple, Dict
import os

# Set font style
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_prediction_scatter(predictions: torch.Tensor,
                          targets: torch.Tensor,
                          title: str = "Prediction vs Target",
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 5),
                          alpha: float = 0.6) -> None:
    """
    Plot prediction vs target scatter plot with performance metrics
    
    Args:
        predictions: Predicted values [batch_size, n_predict, 2]
        targets: True values [batch_size, n_predict, 2]
        title: Plot title
        save_path: Save path
        figsize: Figure size
        alpha: Point transparency
    """
    # Flatten data
    pred_real = predictions[:, :, 0].flatten().numpy()
    pred_imag = predictions[:, :, 1].flatten().numpy()
    target_real = targets[:, :, 0].flatten().numpy()
    target_imag = targets[:, :, 1].flatten().numpy()
    
    # Calculate correlation coefficients
    corr_real = np.corrcoef(target_real, pred_real)[0, 1]
    corr_imag = np.corrcoef(target_imag, pred_imag)[0, 1]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Real part scatter plot
    ax1.scatter(target_real, pred_real, alpha=alpha, s=1, c='blue', label='Data points')
    
    # Perfect prediction line
    real_min = min(target_real.min(), pred_real.min())
    real_max = max(target_real.max(), pred_real.max())
    ax1.plot([real_min, real_max], [real_min, real_max], 'r--', linewidth=2, label='Perfect prediction')
    
    ax1.set_xlabel('True Real Part')
    ax1.set_ylabel('Predicted Real Part')
    ax1.set_title(f'{title} - Real Part\nCorrelation: {corr_real:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Imaginary part scatter plot
    ax2.scatter(target_imag, pred_imag, alpha=alpha, s=1, c='green', label='Data points')
    
    # Perfect prediction line
    imag_min = min(target_imag.min(), pred_imag.min())
    imag_max = max(target_imag.max(), pred_imag.max())
    ax2.plot([imag_min, imag_max], [imag_min, imag_max], 'r--', linewidth=2, label='Perfect prediction')
    
    ax2.set_xlabel('True Imaginary Part')
    ax2.set_ylabel('Predicted Imaginary Part')
    ax2.set_title(f'{title} - Imaginary Part\nCorrelation: {corr_imag:.4f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved: {save_path}")
    
    plt.show()

def plot_complex_magnitude_scatter(predictions: torch.Tensor,
                                 targets: torch.Tensor,
                                 title: str = "Complex Magnitude Prediction",
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (8, 6),
                                 alpha: float = 0.6) -> None:
    """
    Plot complex magnitude scatter plot
    
    Args:
        predictions: Predicted values
        targets: True values
        title: Plot title
        save_path: Save path
        figsize: Figure size
        alpha: Point transparency
    """
    # Convert to complex and calculate magnitude
    pred_complex = predictions[:, :, 0] + 1j * predictions[:, :, 1]
    target_complex = targets[:, :, 0] + 1j * targets[:, :, 1]
    
    pred_magnitude = torch.abs(pred_complex).flatten().numpy()
    target_magnitude = torch.abs(target_complex).flatten().numpy()
    
    # Calculate correlation
    corr_magnitude = np.corrcoef(target_magnitude, pred_magnitude)[0, 1]
    
    # Plot scatter
    plt.figure(figsize=figsize)
    plt.scatter(target_magnitude, pred_magnitude, alpha=alpha, s=1, c='purple')
    
    # Perfect prediction line
    mag_min = min(target_magnitude.min(), pred_magnitude.min())
    mag_max = max(target_magnitude.max(), pred_magnitude.max())
    plt.plot([mag_min, mag_max], [mag_min, mag_max], 'r--', linewidth=2, label='Perfect prediction')
    
    plt.xlabel('True Magnitude')
    plt.ylabel('Predicted Magnitude')
    plt.title(f'{title}\nCorrelation: {corr_magnitude:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Magnitude scatter plot saved: {save_path}")
    
    plt.show()

def plot_phase_scatter(predictions: torch.Tensor,
                      targets: torch.Tensor,
                      title: str = "Phase Prediction",
                      save_path: Optional[str] = None,
                      figsize: Tuple[int, int] = (8, 6),
                      alpha: float = 0.6) -> None:
    """
    Plot phase scatter plot
    
    Args:
        predictions: Predicted values
        targets: True values
        title: Plot title
        save_path: Save path
        figsize: Figure size
        alpha: Point transparency
    """
    # Convert to complex and calculate phase
    pred_complex = predictions[:, :, 0] + 1j * predictions[:, :, 1]
    target_complex = targets[:, :, 0] + 1j * targets[:, :, 1]
    
    pred_phase = torch.angle(pred_complex).flatten().numpy()
    target_phase = torch.angle(target_complex).flatten().numpy()
    
    # Calculate phase correlation (accounting for circular nature)
    phase_diff = np.angle(np.exp(1j * (pred_phase - target_phase)))
    phase_correlation = 1 - np.var(phase_diff) / (2 * np.pi**2)
    
    # Plot scatter
    plt.figure(figsize=figsize)
    plt.scatter(target_phase, pred_phase, alpha=alpha, s=1, c='orange')
    
    # Perfect prediction line
    plt.plot([-np.pi, np.pi], [-np.pi, np.pi], 'r--', linewidth=2, label='Perfect prediction')
    
    plt.xlabel('True Phase (radians)')
    plt.ylabel('Predicted Phase (radians)')
    plt.title(f'{title}\nPhase Correlation: {phase_correlation:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Phase scatter plot saved: {save_path}")
    
    plt.show()

def plot_error_distribution(predictions: torch.Tensor,
                          targets: torch.Tensor,
                          title: str = "Error Distribution",
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 4),
                          bins: int = 50) -> None:
    """
    Plot error distribution histogram
    
    Args:
        predictions: Predicted values
        targets: True values
        title: Plot title
        save_path: Save path
        figsize: Figure size
        bins: Number of histogram bins
    """
    # Calculate errors
    error_real = (predictions[:, :, 0] - targets[:, :, 0]).flatten().numpy()
    error_imag = (predictions[:, :, 1] - targets[:, :, 1]).flatten().numpy()
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Real part error distribution
    ax1.hist(error_real, bins=bins, alpha=0.7, color='blue', density=True)
    ax1.set_xlabel('Real Part Error')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Real Part Error Distribution\nStd: {np.std(error_real):.4f}')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # Imaginary part error distribution
    ax2.hist(error_imag, bins=bins, alpha=0.7, color='green', density=True)
    ax2.set_xlabel('Imaginary Part Error')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Imaginary Part Error Distribution\nStd: {np.std(error_imag):.4f}')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # Magnitude error distribution
    pred_complex = predictions[:, :, 0] + 1j * predictions[:, :, 1]
    target_complex = targets[:, :, 0] + 1j * targets[:, :, 1]
    magnitude_error = (torch.abs(pred_complex) - torch.abs(target_complex)).flatten().numpy()
    
    ax3.hist(magnitude_error, bins=bins, alpha=0.7, color='purple', density=True)
    ax3.set_xlabel('Magnitude Error')
    ax3.set_ylabel('Density')
    ax3.set_title(f'Magnitude Error Distribution\nStd: {np.std(magnitude_error):.4f}')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution plot saved: {save_path}")
    
    plt.show()

def plot_phase_error_comparison(predictions_dict: Dict[str, torch.Tensor],
                              targets: torch.Tensor,
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 6),
                              bins: int = 50) -> None:
    """
    Plot phase error distribution comparison across models
    
    Args:
        predictions_dict: Dictionary of model predictions
        targets: Ground truth targets
        save_path: Save path
        figsize: Figure size
        bins: Number of histogram bins
    """
    plt.figure(figsize=figsize)
    
    colors = {'ResCNN': '#1f77b4', 'Transformer': '#ff7f0e', 'Ensemble': '#2ca02c'}
    
    # Convert targets to complex
    target_complex = targets[:, :, 0] + 1j * targets[:, :, 1]
    target_phase = torch.angle(target_complex).flatten().numpy()
    
    for model_name, predictions in predictions_dict.items():
        # Convert predictions to complex
        pred_complex = predictions[:, :, 0] + 1j * predictions[:, :, 1]
        pred_phase = torch.angle(pred_complex).flatten().numpy()
        
        # Calculate phase error with proper wrapping
        phase_error = pred_phase - target_phase
        phase_error = np.angle(np.exp(1j * phase_error))  # Wrap to [-π, π]
        
        plt.hist(phase_error, bins=bins, alpha=0.6, 
                color=colors.get(model_name, 'black'),
                label=f'{model_name} (std: {np.std(phase_error):.3f})',
                density=True)
    
    plt.xlabel('Phase Error (radians)')
    plt.ylabel('Density')
    plt.title('Phase Error Distribution Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add vertical lines at key points
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    plt.axvline(x=np.pi/4, color='red', linestyle=':', alpha=0.3, linewidth=1)
    plt.axvline(x=-np.pi/4, color='red', linestyle=':', alpha=0.3, linewidth=1)
    
    plt.xlim(-np.pi, np.pi)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Phase error comparison plot saved: {save_path}")
    
    plt.show()

def plot_error_distribution_comparison(predictions_dict: Dict[str, torch.Tensor],
                                     targets: torch.Tensor,
                                     error_type: str = 'real',
                                     save_path: Optional[str] = None,
                                     figsize: Tuple[int, int] = (10, 6),
                                     bins: int = 50) -> None:
    """
    Plot error distribution comparison across models
    
    Args:
        predictions_dict: Dictionary of model predictions
        targets: Ground truth targets
        error_type: Type of error ('real', 'imag', 'magnitude')
        save_path: Save path
        figsize: Figure size
        bins: Number of histogram bins
    """
    plt.figure(figsize=figsize)
    
    colors = {'ResCNN': '#1f77b4', 'Transformer': '#ff7f0e', 'Ensemble': '#2ca02c'}
    
    for model_name, predictions in predictions_dict.items():
        if error_type == 'real':
            error = (predictions[:, :, 0] - targets[:, :, 0]).flatten().numpy()
            xlabel = 'Real Part Error'
        elif error_type == 'imag':
            error = (predictions[:, :, 1] - targets[:, :, 1]).flatten().numpy()
            xlabel = 'Imaginary Part Error'
        elif error_type == 'magnitude':
            pred_complex = predictions[:, :, 0] + 1j * predictions[:, :, 1]
            target_complex = targets[:, :, 0] + 1j * targets[:, :, 1]
            pred_magnitude = torch.abs(pred_complex).flatten().numpy()
            target_magnitude = torch.abs(target_complex).flatten().numpy()
            error = pred_magnitude - target_magnitude
            xlabel = 'Magnitude Error'
        else:
            raise ValueError("error_type must be 'real', 'imag', or 'magnitude'")
        
        plt.hist(error, bins=bins, alpha=0.6,
                color=colors.get(model_name, 'black'),
                label=f'{model_name} (std: {np.std(error):.4f})',
                density=True)
    
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.title(f'{xlabel} Distribution Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add vertical line at zero error
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{error_type.capitalize()} error comparison plot saved: {save_path}")
    
    plt.show()

def plot_antenna_pattern_comparison(predictions: torch.Tensor,
                                  targets: torch.Tensor,
                                  antenna_idx: int = 0,
                                  title: str = "Antenna Pattern Comparison",
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot specific antenna prediction vs truth pattern comparison
    
    Args:
        predictions: Predicted values [batch_size, n_predict, 2]
        targets: True values [batch_size, n_predict, 2]
        antenna_idx: Antenna index
        title: Plot title
        save_path: Save path
        figsize: Figure size
    """
    if antenna_idx >= predictions.size(1):
        print(f"Antenna index {antenna_idx} out of range")
        return
    
    # Extract specific antenna data
    pred_antenna = predictions[:, antenna_idx, :]  # [batch_size, 2]
    target_antenna = targets[:, antenna_idx, :]    # [batch_size, 2]
    
    # Convert to complex
    pred_complex = pred_antenna[:, 0] + 1j * pred_antenna[:, 1]
    target_complex = target_antenna[:, 0] + 1j * target_antenna[:, 1]
    
    # Calculate magnitude and phase
    pred_magnitude = torch.abs(pred_complex).numpy()
    target_magnitude = torch.abs(target_complex).numpy()
    pred_phase = torch.angle(pred_complex).numpy()
    target_phase = torch.angle(target_complex).numpy()
    
    # Create sample indices
    sample_indices = np.arange(len(pred_magnitude))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Magnitude comparison
    ax1.plot(sample_indices, target_magnitude, 'b-', label='True', alpha=0.7)
    ax1.plot(sample_indices, pred_magnitude, 'r--', label='Predicted', alpha=0.7)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Magnitude')
    ax1.set_title(f'Antenna {antenna_idx} - Magnitude Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Phase comparison
    ax2.plot(sample_indices, target_phase, 'b-', label='True', alpha=0.7)
    ax2.plot(sample_indices, pred_phase, 'r--', label='Predicted', alpha=0.7)
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Phase (radians)')
    ax2.set_title(f'Antenna {antenna_idx} - Phase Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-np.pi, np.pi])
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Antenna pattern comparison plot saved: {save_path}")
    
    plt.show()

def create_all_scatter_plots(results_dict: dict, save_dir: str = "results") -> None:
    """
    Create scatter plots for all models
    
    Args:
        results_dict: Dictionary containing model results
        save_dir: Save directory
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data for comparison plots
    predictions_dict = {}
    targets = None
    
    for model_name, result in results_dict.items():
        predictions = result['predictions']
        current_targets = result['targets']
        
        # Individual model plots
        save_path = os.path.join(save_dir, f"scatter_{model_name.lower()}.png")
        plot_prediction_scatter(predictions, current_targets, 
                              title=f"{model_name} Model Prediction",
                              save_path=save_path)
        
        # Magnitude scatter plot
        magnitude_save_path = os.path.join(save_dir, f"magnitude_scatter_{model_name.lower()}.png")
        plot_complex_magnitude_scatter(predictions, current_targets,
                                     title=f"{model_name} Model - Magnitude",
                                     save_path=magnitude_save_path)
        
        # Phase scatter plot
        phase_save_path = os.path.join(save_dir, f"phase_scatter_{model_name.lower()}.png")
        plot_phase_scatter(predictions, current_targets,
                          title=f"{model_name} Model - Phase",
                          save_path=phase_save_path)
        
        # Individual error distribution
        error_save_path = os.path.join(save_dir, f"error_distribution_{model_name.lower()}.png")
        plot_error_distribution(predictions, current_targets,
                               title=f"{model_name} Model - Error Distribution",
                               save_path=error_save_path)
        
        # Store for comparison plots
        predictions_dict[model_name] = predictions
        if targets is None:
            targets = current_targets
    
    # Comparison plots across all models
    if len(predictions_dict) > 1:
        # Phase error comparison
        plot_phase_error_comparison(
            predictions_dict, targets,
            save_path=os.path.join(save_dir, "phase_error_comparison.png")
        )
        
        # Error distribution comparisons
        for error_type in ['real', 'imag', 'magnitude']:
            plot_error_distribution_comparison(
                predictions_dict, targets,
                error_type=error_type,
                save_path=os.path.join(save_dir, f"{error_type}_error_comparison.png")
            )

if __name__ == "__main__":
    # Test plotting functions
    print("Testing scatter plot visualization functions...")
    
    # Generate test data
    batch_size, n_predict = 1000, 48
    targets = torch.randn(batch_size, n_predict, 2)
    predictions = targets + 0.1 * torch.randn_like(targets)
    
    # Test plots
    plot_prediction_scatter(predictions, targets, title="Test Model")
    plot_complex_magnitude_scatter(predictions, targets, title="Test Magnitude")
    plot_phase_scatter(predictions, targets, title="Test Phase")
    plot_error_distribution(predictions, targets, title="Test Error Distribution")
    plot_antenna_pattern_comparison(predictions, targets, antenna_idx=0)
    
    print("Scatter plot visualization test completed!")