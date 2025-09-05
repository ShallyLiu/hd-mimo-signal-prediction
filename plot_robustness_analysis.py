"""
Robustness Analysis Visualization Module
Plotting functions for SNR robustness and scalability experiments
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, Tuple
import os

# Set plotting style
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

def plot_snr_robustness(snr_range: np.ndarray,
                       nmse_results: Dict[str, np.ndarray],
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (10, 6)) -> None:
    """Plot SNR robustness curve"""
    plt.figure(figsize=figsize)
    
    # Convert NMSE to dB
    colors = {'ResCNN': '#1f77b4', 'Transformer': '#ff7f0e', 'AdaptiveEnsemble': '#2ca02c'}
    markers = {'ResCNN': 'o', 'Transformer': 's', 'AdaptiveEnsemble': '^'}
    linestyles = {'ResCNN': '-', 'Transformer': '--', 'AdaptiveEnsemble': '-'}
    
    for model_name, nmse_values in nmse_results.items():
        if model_name in ['ResCNN', 'Transformer', 'AdaptiveEnsemble']:
            nmse_db = 10 * np.log10(nmse_values + 1e-12)
            plt.plot(snr_range, nmse_db, 
                    color=colors.get(model_name, 'black'),
                    marker=markers.get(model_name, 'o'),
                    linestyle=linestyles.get(model_name, '-'),
                    linewidth=2.5,
                    markersize=6,
                    markerfacecolor='white',
                    markeredgewidth=2,
                    label=model_name)
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('NMSE (dB)')
    plt.title('SNR Robustness Analysis')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Add performance annotation for key SNR point
    if 'AdaptiveEnsemble' in nmse_results:
        key_snr = 10  # 10 dB SNR
        if key_snr in snr_range:
            idx = np.where(snr_range == key_snr)[0][0]
            ensemble_nmse_db = 10 * np.log10(nmse_results['AdaptiveEnsemble'][idx] + 1e-12)
            plt.annotate(f'Adaptive: {ensemble_nmse_db:.1f} dB',
                        xy=(key_snr, ensemble_nmse_db),
                        xytext=(key_snr + 3, ensemble_nmse_db - 1),
                        arrowprops=dict(arrowstyle='->', color='#2ca02c', alpha=0.7),
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.xlim(snr_range[0], snr_range[-1])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SNR robustness plot saved: {save_path}")
    
    plt.show()

def plot_antenna_spacing_analysis(spacing_factors: np.ndarray,
                                 nmse_results: Dict[float, Dict[str, float]],
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (10, 6)) -> None:
    """Plot antenna spacing analysis curve"""
    plt.figure(figsize=figsize)
    
    # Organize results by model
    model_results = {}
    for spacing in spacing_factors:
        for model_name, nmse in nmse_results[spacing].items():
            if model_name in ['ResCNN', 'Transformer', 'AdaptiveEnsemble']:
                if model_name not in model_results:
                    model_results[model_name] = []
                model_results[model_name].append(nmse)
    
    colors = {'ResCNN': '#1f77b4', 'Transformer': '#ff7f0e', 'AdaptiveEnsemble': '#2ca02c'}
    markers = {'ResCNN': 'o', 'Transformer': 's', 'AdaptiveEnsemble': '^'}
    linestyles = {'ResCNN': '-', 'Transformer': '--', 'AdaptiveEnsemble': '-'}
    
    for model_name, nmse_values in model_results.items():
        nmse_db = 10 * np.log10(np.array(nmse_values) + 1e-12)
        plt.plot(spacing_factors, nmse_db,
                color=colors.get(model_name, 'black'),
                marker=markers.get(model_name, 'o'),
                linestyle=linestyles.get(model_name, '-'),
                linewidth=2.5,
                markersize=6,
                markerfacecolor='white',
                markeredgewidth=2,
                label=model_name)
    
    plt.xlabel('Antenna Spacing (λ/2)')
    plt.ylabel('NMSE (dB)')
    plt.title('Antenna Spacing Scalability Analysis')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Add annotation for the trained configuration
    trained_spacing = 0.5  # λ/2
    if trained_spacing in spacing_factors:
        plt.axvline(x=trained_spacing, color='red', linestyle=':', alpha=0.5, linewidth=2)
        plt.text(trained_spacing + 0.01, plt.ylim()[1] - 0.5, 'Training Configuration',
                rotation=90, fontsize=10, color='red', alpha=0.7)
    
    plt.xlim(spacing_factors[0], spacing_factors[-1])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Antenna spacing analysis plot saved: {save_path}")
    
    plt.show()

def plot_spatial_error_distribution(per_antenna_nmse: Dict[str, np.ndarray],
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (12, 6)) -> None:
    """Plot per-antenna spatial error distribution"""
    plt.figure(figsize=figsize)
    
    antenna_indices = np.arange(1, 49)  # 48 predicted antennas
    colors = {'ResCNN': '#1f77b4', 'Transformer': '#ff7f0e', 'AdaptiveEnsemble': '#2ca02c'}
    
    for model_name, nmse_values in per_antenna_nmse.items():
        if model_name in ['ResCNN', 'Transformer', 'AdaptiveEnsemble']:
            nmse_db = 10 * np.log10(nmse_values + 1e-12)
            plt.plot(antenna_indices, nmse_db,
                    color=colors.get(model_name, 'black'),
                    linewidth=2,
                    alpha=0.8,
                    label=model_name)
    
    plt.xlabel('Antenna Index')
    plt.ylabel('NMSE (dB)')
    plt.title('Spatial Error Distribution')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Add average performance lines
    for model_name, nmse_values in per_antenna_nmse.items():
        if model_name in ['ResCNN', 'Transformer', 'AdaptiveEnsemble']:
            avg_nmse_db = 10 * np.log10(np.mean(nmse_values) + 1e-12)
            plt.axhline(y=avg_nmse_db, color=colors.get(model_name, 'black'),
                       linestyle='--', alpha=0.5, linewidth=1)
    
    plt.xlim(1, 48)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spatial error distribution plot saved: {save_path}")
    
    plt.show()

def plot_spatial_error_heatmap(per_antenna_nmse: Dict[str, np.ndarray],
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (15, 4)) -> None:
    """Plot spatial error as heatmap (8x6 antenna grid for 48 antennas)"""
    model_names = [name for name in per_antenna_nmse.keys() if name in ['ResCNN', 'Transformer', 'AdaptiveEnsemble']]
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    
    # Reshape 48 antennas to 8x6 grid
    for i, model_name in enumerate(model_names):
        nmse_values = per_antenna_nmse[model_name]
        nmse_db = 10 * np.log10(nmse_values + 1e-12)
        
        # Reshape to grid (8x6 = 48)
        nmse_grid = nmse_db.reshape(8, 6)
        
        im = axes[i].imshow(nmse_grid, cmap='viridis', aspect='auto')
        axes[i].set_title(f'{model_name}')
        axes[i].set_xlabel('Antenna Column')
        axes[i].set_ylabel('Antenna Row')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[i])
        cbar.set_label('NMSE (dB)')
        
        # Add antenna indices as text
        for row in range(8):
            for col in range(6):
                antenna_idx = row * 6 + col + 1
                axes[i].text(col, row, str(antenna_idx), 
                           ha='center', va='center', color='white', fontsize=8)
    
    plt.suptitle('Spatial Error Heatmap')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spatial error heatmap saved: {save_path}")
    
    plt.show()

def plot_phase_error_distribution(predictions_dict: Dict[str, np.ndarray],
                                 targets: np.ndarray,
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (10, 6),
                                 bins: int = 50) -> None:
    """Plot phase error distribution comparison"""
    plt.figure(figsize=figsize)
    
    colors = {'ResCNN': '#1f77b4', 'Transformer': '#ff7f0e', 'AdaptiveEnsemble': '#2ca02c'}
    
    # Convert targets to complex
    target_complex = targets[:, :, 0] + 1j * targets[:, :, 1]
    target_phase = np.angle(target_complex).flatten()
    
    for model_name, predictions in predictions_dict.items():
        if model_name in ['ResCNN', 'Transformer', 'AdaptiveEnsemble']:
            # Convert predictions to complex
            pred_complex = predictions[:, :, 0] + 1j * predictions[:, :, 1]
            pred_phase = np.angle(pred_complex).flatten()
            
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
        print(f"Phase error distribution plot saved: {save_path}")
    
    plt.show()

def plot_error_comparison_histogram(predictions_dict: Dict[str, np.ndarray],
                                   targets: np.ndarray,
                                   error_type: str = 'real',
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (10, 6),
                                   bins: int = 50) -> None:
    """Plot error distribution comparison histogram"""
    plt.figure(figsize=figsize)
    
    colors = {'ResCNN': '#1f77b4', 'Transformer': '#ff7f0e', 'AdaptiveEnsemble': '#2ca02c'}
    
    for model_name, predictions in predictions_dict.items():
        if model_name in ['ResCNN', 'Transformer', 'AdaptiveEnsemble']:
            if error_type == 'real':
                error = (predictions[:, :, 0] - targets[:, :, 0]).flatten()
                xlabel = 'Real Part Error'
            elif error_type == 'imag':
                error = (predictions[:, :, 1] - targets[:, :, 1]).flatten()
                xlabel = 'Imaginary Part Error'
            elif error_type == 'magnitude':
                pred_complex = predictions[:, :, 0] + 1j * predictions[:, :, 1]
                target_complex = targets[:, :, 0] + 1j * targets[:, :, 1]
                error = (np.abs(pred_complex) - np.abs(target_complex)).flatten()
                xlabel = 'Magnitude Error'
            else:
                raise ValueError("error_type must be 'real', 'imag', or 'magnitude'")
            
            plt.hist(error, bins=bins, alpha=0.6,
                    color=colors.get(model_name, 'black'),
                    label=f'{model_name} (std: {np.std(error):.3f})',
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

def create_all_robustness_plots(results_dict: Dict, 
                               predictions_dict: Dict[str, np.ndarray],
                               targets: np.ndarray,
                               save_dir: str = "results") -> None:
    """Create all robustness analysis plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    # SNR robustness plot
    if 'snr_robustness' in results_dict:
        snr_data = results_dict['snr_robustness']
        plot_snr_robustness(
            snr_data['snr_range'],
            snr_data['nmse_results'],
            save_path=os.path.join(save_dir, "snr_robustness.png")
        )
    
    # Antenna spacing analysis plot
    if 'antenna_spacing' in results_dict:
        spacing_data = results_dict['antenna_spacing']
        plot_antenna_spacing_analysis(
            spacing_data['spacing_factors'],
            spacing_data['nmse_results'],
            save_path=os.path.join(save_dir, "antenna_spacing_analysis.png")
        )
    
    # Spatial error plots
    if 'spatial_error' in results_dict:
        plot_spatial_error_distribution(
            results_dict['spatial_error'],
            save_path=os.path.join(save_dir, "spatial_error_distribution.png")
        )
        
        plot_spatial_error_heatmap(
            results_dict['spatial_error'],
            save_path=os.path.join(save_dir, "spatial_error_heatmap.png")
        )
    
    # Phase error distribution (only if predictions are available)
    if predictions_dict and targets is not None:
        filtered_predictions = {k: v for k, v in predictions_dict.items() 
                              if k in ['ResCNN', 'Transformer', 'AdaptiveEnsemble']}
        if filtered_predictions:
            plot_phase_error_distribution(
                filtered_predictions,
                targets,
                save_path=os.path.join(save_dir, "phase_error_distribution.png")
            )
            
            # Error comparison histograms
            for error_type in ['real', 'imag', 'magnitude']:
                plot_error_comparison_histogram(
                    filtered_predictions,
                    targets,
                    error_type=error_type,
                    save_path=os.path.join(save_dir, f"{error_type}_error_comparison.png")
                )

if __name__ == "__main__":
    # Test plotting functions with dummy data
    print("Testing robustness visualization functions...")
    
    # Dummy SNR robustness data
    snr_range = np.arange(-10, 21, 2)
    nmse_results = {
        'ResCNN': 0.3 * np.exp(-snr_range/15) + 0.05,
        'Transformer': 0.35 * np.exp(-snr_range/15) + 0.06,
        'AdaptiveEnsemble': 0.25 * np.exp(-snr_range/15) + 0.04
    }
    plot_snr_robustness(snr_range, nmse_results)
    
    print("Robustness visualization test completed!")