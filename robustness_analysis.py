"""
Robustness Analysis Module
SNR robustness and scalability experiments for MIMO models
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os
from data_simulator import MIMODataSimulator
from models.rescnn import ResCNN
from models.transformer import MIMOTransformer
from ensemble.fusion import ensemble_output, EnsembleOptimizer, AdaptiveEnsemble
from evaluate.metrics import compute_nmse
from train.utils import create_data_loader

class RobustnessAnalyzer:
    """Robustness and scalability analyzer for MIMO models"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.results = {}
        self.adaptive_ensemble = None
        
    def load_trained_models(self, n_observed: int = 16, n_predict: int = 48) -> Dict[str, torch.nn.Module]:
        """Load pre-trained models"""
        models = {}
        
        # Load ResCNN
        rescnn_model = ResCNN(
            input_dim=3, n_observed=n_observed, n_predict=n_predict,
            hidden_dim=128, n_layers=6
        ).to(self.device)
        
        if os.path.exists("checkpoints/rescnn_best.pth"):
            checkpoint = torch.load("checkpoints/rescnn_best.pth", map_location=self.device)
            rescnn_model.load_state_dict(checkpoint['model_state_dict'])
            models['ResCNN'] = rescnn_model
            print("Loaded ResCNN model")
        
        # Load Transformer
        transformer_model = MIMOTransformer(
            input_dim=3, n_observed=n_observed, n_predict=n_predict,
            d_model=96, n_heads=3, n_layers=2, d_ff=192, dropout=0.1
        ).to(self.device)
        
        if os.path.exists("checkpoints/transformer_best.pth"):
            checkpoint = torch.load("checkpoints/transformer_best.pth", map_location=self.device)
            transformer_model.load_state_dict(checkpoint['model_state_dict'])
            models['Transformer'] = transformer_model
            print("Loaded Transformer model")
        
        return models
    
    def train_adaptive_ensemble(self, 
                              models: Dict[str, torch.nn.Module],
                              feature_type: str = 'snr',
                              n_train_samples: int = 1000) -> AdaptiveEnsemble:
        """
        Train adaptive ensemble module
        
        Args:
            models: Dictionary of trained models
            feature_type: Type of input feature ('snr' or 'spacing')
            n_train_samples: Number of training samples
        """
        print(f"Training adaptive ensemble for {feature_type} features...")
        
        # Create adaptive ensemble
        input_dim = 1  # SNR or spacing factor
        adaptive_ensemble = AdaptiveEnsemble(
            input_dim=input_dim, n_models=2, hidden_dim=32
        ).to(self.device)
        
        # Generate different conditions for training
        if feature_type == 'snr':
            # Create different SNR levels
            snr_levels = [-5, 0, 5, 10, 15]
            samples_per_condition = n_train_samples // len(snr_levels)
        else:  # spacing
            # Create different spacing factors
            spacing_levels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            samples_per_condition = n_train_samples // len(spacing_levels)
        
        # Collect training data
        all_features = []
        all_rescnn_outputs = []
        all_transformer_outputs = []
        all_targets = []
        
        if feature_type == 'snr':
            for snr in snr_levels:
                print(f"  Generating data for SNR = {snr} dB...")
                noise_std = np.sqrt(10 ** (-snr / 10))
                
                simulator = MIMODataSimulator(
                    n_antennas=64, n_observed=16, n_predict=48,
                    noise_std=noise_std
                )
                X_batch, Y_batch = simulator.generate_training_data(samples_per_condition)
                
                # Convert to tensors
                X_tensor = torch.from_numpy(X_batch).float().to(self.device)
                Y_tensor = torch.from_numpy(Y_batch).float().to(self.device)
                
                # Get model predictions
                with torch.no_grad():
                    rescnn_pred = models['ResCNN'](X_tensor)
                    transformer_pred = models['Transformer'](X_tensor)
                
                # Store features and predictions
                condition_features = [snr] * samples_per_condition
                all_features.extend(condition_features)
                all_rescnn_outputs.append(rescnn_pred)
                all_transformer_outputs.append(transformer_pred)
                all_targets.append(Y_tensor)
        else:  # spacing
            for spacing_factor in spacing_levels:
                print(f"  Generating data for spacing = {spacing_factor} λ/2...")
                wavelength = 3e8 / 3.5e9 / spacing_factor
                
                simulator = MIMODataSimulator(
                    n_antennas=64, n_observed=16, n_predict=48,
                    noise_std=0.1, wavelength=wavelength
                )
                X_batch, Y_batch = simulator.generate_training_data(samples_per_condition)
                
                # Convert to tensors
                X_tensor = torch.from_numpy(X_batch).float().to(self.device)
                Y_tensor = torch.from_numpy(Y_batch).float().to(self.device)
                
                # Get model predictions
                with torch.no_grad():
                    rescnn_pred = models['ResCNN'](X_tensor)
                    transformer_pred = models['Transformer'](X_tensor)
                
                # Store features and predictions
                condition_features = [spacing_factor] * samples_per_condition
                all_features.extend(condition_features)
                all_rescnn_outputs.append(rescnn_pred)
                all_transformer_outputs.append(transformer_pred)
                all_targets.append(Y_tensor)
        
        # Convert to tensors
        features_tensor = torch.tensor(all_features, dtype=torch.float32).unsqueeze(1).to(self.device)
        rescnn_tensor = torch.cat(all_rescnn_outputs, dim=0)
        transformer_tensor = torch.cat(all_transformer_outputs, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)
        
        # Create dataset and dataloader
        dataset = TensorDataset(features_tensor, rescnn_tensor, transformer_tensor, targets_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train adaptive ensemble
        optimizer = optim.AdamW(adaptive_ensemble.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        adaptive_ensemble.train()
        for epoch in range(50):
            epoch_loss = 0.0
            for features, rescnn_outputs, transformer_outputs, targets in dataloader:
                optimizer.zero_grad()
                
                ensemble_output, weights = adaptive_ensemble(
                    features, [rescnn_outputs, transformer_outputs]
                )
                
                loss = criterion(ensemble_output, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/50, Loss: {epoch_loss/len(dataloader):.6f}")
        
        adaptive_ensemble.eval()
        return adaptive_ensemble
    
    def snr_robustness_experiment(self, 
                                 snr_range: np.ndarray = None,
                                 n_test_samples: int = 1000) -> Dict[str, np.ndarray]:
        """SNR robustness experiment with adaptive ensemble"""
        if snr_range is None:
            snr_range = np.arange(-10, 21, 2)  # -10 to 20 dB
        
        # Convert SNR (dB) to noise standard deviation
        noise_std_range = np.sqrt(10 ** (-snr_range / 10))
        
        models = self.load_trained_models()
        results = {model_name: [] for model_name in models.keys()}
        results['Ensemble'] = []
        results['AdaptiveEnsemble'] = []
        
        print(f"Running SNR robustness experiment with {len(snr_range)} SNR points...")
        
        # Train adaptive ensemble for SNR
        if len(models) == 2:
            adaptive_ensemble = self.train_adaptive_ensemble(models, 'snr')
        
        for i, (snr, noise_std) in enumerate(zip(snr_range, noise_std_range)):
            print(f"Testing SNR = {snr} dB (noise_std = {noise_std:.4f}) [{i+1}/{len(snr_range)}]")
            
            # Generate test data with specific noise level
            simulator = MIMODataSimulator(
                n_antennas=64, n_observed=16, n_predict=48,
                noise_std=noise_std
            )
            X_test, Y_test = simulator.generate_training_data(n_test_samples)
            
            test_loader = create_data_loader(
                X_test, Y_test, batch_size=64, shuffle=False, num_workers=0
            )
            
            model_predictions = {}
            
            # Test each model
            for model_name, model in models.items():
                model.eval()
                predictions = []
                targets = []
                
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data)
                        predictions.append(output.cpu())
                        targets.append(target.cpu())
                
                pred_tensor = torch.cat(predictions, dim=0)
                target_tensor = torch.cat(targets, dim=0)
                
                nmse = compute_nmse(pred_tensor, target_tensor)
                results[model_name].append(nmse)
                model_predictions[model_name] = pred_tensor
            
            # Test fixed ensemble
            if 'ResCNN' in model_predictions and 'Transformer' in model_predictions:
                ensemble_pred = ensemble_output(
                    model_predictions['ResCNN'], 
                    model_predictions['Transformer'], 
                    alpha=0.6
                )
                nmse = compute_nmse(ensemble_pred, target_tensor)
                results['Ensemble'].append(nmse)
                
                # Test adaptive ensemble
                snr_features = torch.full((len(target_tensor),), snr, dtype=torch.float32).unsqueeze(1).to(self.device)
                rescnn_preds_device = model_predictions['ResCNN'].to(self.device)
                transformer_preds_device = model_predictions['Transformer'].to(self.device)
                
                with torch.no_grad():
                    adaptive_pred, _ = adaptive_ensemble(
                        snr_features, [rescnn_preds_device, transformer_preds_device]
                    )
                    adaptive_pred = adaptive_pred.cpu()
                
                nmse = compute_nmse(adaptive_pred, target_tensor)
                results['AdaptiveEnsemble'].append(nmse)
        
        # Convert to numpy arrays
        for key in results:
            results[key] = np.array(results[key])
        
        self.results['snr_robustness'] = {
            'snr_range': snr_range,
            'nmse_results': results
        }
        
        return results
    
    def antenna_spacing_experiment(self, 
                                  spacing_factors: np.ndarray = None,
                                  n_test_samples: int = 1000) -> Dict[str, np.ndarray]:
        """
        Antenna spacing scalability experiment with adaptive ensemble
        """
        if spacing_factors is None:
            spacing_factors = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])  # multiples of λ/2
        
        models = self.load_trained_models(n_observed=16, n_predict=48)
        results = {}
        
        print(f"Running antenna spacing experiment with {len(spacing_factors)} spacing factors...")
        
        # Train adaptive ensemble for spacing
        if len(models) == 2:
            adaptive_ensemble = self.train_adaptive_ensemble(models, 'spacing')
        
        for spacing_factor in spacing_factors:
            print(f"Testing spacing factor = {spacing_factor} λ/2")
            
            # Create simulator with modified wavelength
            modified_wavelength = 3e8 / 3.5e9 / spacing_factor
            
            simulator = MIMODataSimulator(
                n_antennas=64, n_observed=16, n_predict=48,
                noise_std=0.1, wavelength=modified_wavelength
            )
            X_test, Y_test = simulator.generate_training_data(n_test_samples)
            
            test_loader = create_data_loader(
                X_test, Y_test, batch_size=64, shuffle=False, num_workers=0
            )
            
            spacing_results = {}
            model_predictions = {}
            
            # Test each model
            for model_name, model in models.items():
                model.eval()
                predictions = []
                targets = []
                
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data)
                        predictions.append(output.cpu())
                        targets.append(target.cpu())
                
                pred_tensor = torch.cat(predictions, dim=0)
                target_tensor = torch.cat(targets, dim=0)
                
                nmse = compute_nmse(pred_tensor, target_tensor)
                spacing_results[model_name] = nmse
                model_predictions[model_name] = pred_tensor
            
            # Test fixed ensemble
            if 'ResCNN' in model_predictions and 'Transformer' in model_predictions:
                ensemble_pred = ensemble_output(
                    model_predictions['ResCNN'], 
                    model_predictions['Transformer'], 
                    alpha=0.6
                )
                nmse = compute_nmse(ensemble_pred, target_tensor)
                spacing_results['Ensemble'] = nmse
                
                # Test adaptive ensemble
                spacing_features = torch.full((len(target_tensor),), spacing_factor, dtype=torch.float32).unsqueeze(1).to(self.device)
                rescnn_preds_device = model_predictions['ResCNN'].to(self.device)
                transformer_preds_device = model_predictions['Transformer'].to(self.device)
                
                with torch.no_grad():
                    adaptive_pred, _ = adaptive_ensemble(
                        spacing_features, [rescnn_preds_device, transformer_preds_device]
                    )
                    adaptive_pred = adaptive_pred.cpu()
                
                nmse = compute_nmse(adaptive_pred, target_tensor)
                spacing_results['AdaptiveEnsemble'] = nmse
            
            results[spacing_factor] = spacing_results
        
        self.results['antenna_spacing'] = {
            'spacing_factors': spacing_factors,
            'nmse_results': results
        }
        
        return results
    
    def spatial_error_analysis(self, n_test_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Per-antenna spatial error analysis with adaptive ensemble"""
        print("Running spatial error analysis...")
        
        models = self.load_trained_models()
        
        # Generate test data
        simulator = MIMODataSimulator(
            n_antennas=64, n_observed=16, n_predict=48, noise_std=0.1
        )
        X_test, Y_test = simulator.generate_training_data(n_test_samples)
        
        test_loader = create_data_loader(
            X_test, Y_test, batch_size=64, shuffle=False, num_workers=0
        )
        
        results = {}
        all_predictions = {}
        all_targets = None
        
        # Train adaptive ensemble for fixed conditions (using SNR=10dB as reference)
        if len(models) == 2:
            adaptive_ensemble = self.train_adaptive_ensemble(models, 'snr')
        
        # Get predictions from all models
        for model_name, model in models.items():
            model.eval()
            predictions = []
            targets = []
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    predictions.append(output.cpu())
                    targets.append(target.cpu())
            
            pred_tensor = torch.cat(predictions, dim=0)
            target_tensor = torch.cat(targets, dim=0)
            
            all_predictions[model_name] = pred_tensor
            if all_targets is None:
                all_targets = target_tensor
        
        # Add fixed ensemble predictions
        if 'ResCNN' in all_predictions and 'Transformer' in all_predictions:
            ensemble_pred = ensemble_output(
                all_predictions['ResCNN'], 
                all_predictions['Transformer'], 
                alpha=0.6
            )
            all_predictions['Ensemble'] = ensemble_pred
            
            # Add adaptive ensemble predictions
            reference_snr = 10.0  # reference SNR for spatial analysis
            snr_features = torch.full((len(all_targets),), reference_snr, dtype=torch.float32).unsqueeze(1).to(self.device)
            rescnn_preds_device = all_predictions['ResCNN'].to(self.device)
            transformer_preds_device = all_predictions['Transformer'].to(self.device)
            
            with torch.no_grad():
                adaptive_pred, _ = adaptive_ensemble(
                    snr_features, [rescnn_preds_device, transformer_preds_device]
                )
                adaptive_pred = adaptive_pred.cpu()
            
            all_predictions['AdaptiveEnsemble'] = adaptive_pred
        
        # Calculate per-antenna NMSE
        for model_name, predictions in all_predictions.items():
            per_antenna_nmse = []
            
            for antenna_idx in range(predictions.shape[1]):  # 48 antennas
                pred_antenna = predictions[:, antenna_idx, :]
                target_antenna = all_targets[:, antenna_idx, :]
                
                # Convert to complex
                pred_complex = pred_antenna[:, 0] + 1j * pred_antenna[:, 1]
                target_complex = target_antenna[:, 0] + 1j * target_antenna[:, 1]
                
                # Calculate NMSE for this antenna
                numerator = torch.mean(torch.abs(pred_complex - target_complex) ** 2)
                denominator = torch.mean(torch.abs(target_complex) ** 2)
                nmse = (numerator / (denominator + 1e-8)).item()
                
                per_antenna_nmse.append(nmse)
            
            results[model_name] = np.array(per_antenna_nmse)
        
        self.results['spatial_error'] = results
        return results
    
    def run_all_experiments(self) -> Dict:
        """Run all robustness and scalability experiments with adaptive ensemble"""
        print("Starting comprehensive robustness and scalability analysis with adaptive ensemble...")
        
        self.snr_robustness_experiment()
        self.antenna_spacing_experiment()
        self.spatial_error_analysis()
        
        print("All experiments completed!")
        return self.results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyzer = RobustnessAnalyzer(device)
    
    results = analyzer.run_all_experiments()
    print("Robustness analysis with adaptive ensemble completed!")
    
    # Import visualization functions from evaluate subdirectory
    try:
        from evaluate.plot_robustness_analysis import create_all_robustness_plots
        
        print("\nGenerating robustness analysis plots...")
        # Create plots with the results
        create_all_robustness_plots(
            results_dict=results,
            predictions_dict={},  # Empty since we don't need prediction comparison plots
            targets=None,
            save_dir="results"
        )
        print("All robustness plots generated and saved to 'results' directory!")
        
    except ImportError as e:
        print(f"Could not import visualization module: {e}")
        print("Please ensure plot_robustness_analysis.py is in the evaluate directory.")
    except Exception as e:
        print(f"Error generating plots: {e}")
        print("Analysis results are still available in the analyzer.results dictionary.")