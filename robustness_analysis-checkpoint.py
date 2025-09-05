"""
Robustness Analysis Module
SNR robustness and scalability experiments for MIMO models
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os
from data_simulator import MIMODataSimulator
from models.rescnn import ResCNN
from models.transformer import MIMOTransformer
from ensemble.fusion import ensemble_output, EnsembleOptimizer
from evaluate.metrics import compute_nmse
from train.utils import create_data_loader

class RobustnessAnalyzer:
    """Robustness and scalability analyzer for MIMO models"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.results = {}
        
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
    
    def snr_robustness_experiment(self, 
                                 snr_range: np.ndarray = None,
                                 n_test_samples: int = 1000) -> Dict[str, np.ndarray]:
        """SNR robustness experiment"""
        if snr_range is None:
            snr_range = np.arange(-10, 21, 2)  # -10 to 20 dB
        
        # Convert SNR (dB) to noise standard deviation
        noise_std_range = np.sqrt(10 ** (-snr_range / 10))
        
        models = self.load_trained_models()
        results = {model_name: [] for model_name in models.keys()}
        results['Ensemble'] = []
        
        print(f"Running SNR robustness experiment with {len(snr_range)} SNR points...")
        
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
            
            # Test ensemble if both models are available
            if 'ResCNN' in model_predictions and 'Transformer' in model_predictions:
                ensemble_pred = ensemble_output(
                    model_predictions['ResCNN'], 
                    model_predictions['Transformer'], 
                    alpha=0.6
                )
                nmse = compute_nmse(ensemble_pred, target_tensor)
                results['Ensemble'].append(nmse)
        
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
        Antenna spacing scalability experiment
        Tests model performance with different antenna spacings while keeping 16-48 configuration
        """
        if spacing_factors is None:
            spacing_factors = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])  # multiples of λ/2
        
        models = self.load_trained_models(n_observed=16, n_predict=48)
        results = {}
        
        print(f"Running antenna spacing experiment with {len(spacing_factors)} spacing factors...")
        
        for spacing_factor in spacing_factors:
            print(f"Testing spacing factor = {spacing_factor} λ/2")
            
            # Create simulator with modified wavelength (effectively changing spacing)
            # Original training was done with λ/2 spacing (spacing_factor = 0.5)
            modified_wavelength = 3e8 / 3.5e9 / spacing_factor  # Adjust wavelength to change effective spacing
            
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
            
            # Test ensemble
            if 'ResCNN' in model_predictions and 'Transformer' in model_predictions:
                ensemble_pred = ensemble_output(
                    model_predictions['ResCNN'], 
                    model_predictions['Transformer'], 
                    alpha=0.6
                )
                nmse = compute_nmse(ensemble_pred, target_tensor)
                spacing_results['Ensemble'] = nmse
            
            results[spacing_factor] = spacing_results
        
        self.results['antenna_spacing'] = {
            'spacing_factors': spacing_factors,
            'nmse_results': results
        }
        
        return results
    
    def spatial_error_analysis(self, n_test_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Per-antenna spatial error analysis"""
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
        
        # Add ensemble predictions
        if 'ResCNN' in all_predictions and 'Transformer' in all_predictions:
            ensemble_pred = ensemble_output(
                all_predictions['ResCNN'], 
                all_predictions['Transformer'], 
                alpha=0.6
            )
            all_predictions['Ensemble'] = ensemble_pred
        
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
        """Run all robustness and scalability experiments"""
        print("Starting comprehensive robustness and scalability analysis...")
        
        self.snr_robustness_experiment()
        self.antenna_spacing_experiment()  # Changed from scalability_experiment
        self.spatial_error_analysis()
        
        print("All experiments completed!")
        return self.results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyzer = RobustnessAnalyzer(device)
    
    results = analyzer.run_all_experiments()
    print("Robustness analysis completed!")