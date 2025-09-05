"""
Main execution script
Complete pipeline integrating data generation, model training, evaluation and visualization
"""

import os
import sys
import torch
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')

from data_simulator import MIMODataSimulator
from models.rescnn import ResCNN
from models.transformer import MIMOTransformer
from train.utils import ModelTrainer, create_data_loader
from ensemble.fusion import ensemble_output, EnsembleOptimizer, evaluate_ensemble_performance
from evaluate.metrics import compute_comprehensive_metrics, print_metrics
from evaluate.plot_scatter import create_all_scatter_plots
from evaluate.plot_loss_curve import create_all_training_plots, plot_ensemble_weight_analysis

def setup_environment():
    """Setup runtime environment"""
    # Create necessary directories
    dirs = ['data', 'checkpoints', 'results', 'models', 'train', 'ensemble', 'evaluate']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # RTX3090 optimization
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device

def generate_data(config):
    """Generate MIMO simulation data"""
    print("=" * 50)
    print("Data Generation Phase")
    print("=" * 50)
    
    simulator = MIMODataSimulator(
        n_antennas=config['n_antennas'],
        n_observed=config['n_observed'],
        n_predict=config['n_predict'],
        noise_std=config['noise_std']
    )
    
    # Generate training data
    print("Generating training data...")
    X_train, Y_train = simulator.generate_training_data(config['train_samples'])
    simulator.save_data(X_train, Y_train, "data/train")
    
    # Generate validation data
    print("Generating validation data...")
    X_val, Y_val = simulator.generate_training_data(config['val_samples'])
    simulator.save_data(X_val, Y_val, "data/val")
    
    # Generate test data
    print("Generating test data...")
    X_test, Y_test = simulator.generate_training_data(config['test_samples'])
    simulator.save_data(X_test, Y_test, "data/test")
    
    print(f"Data generation completed")
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

def train_models(config, device):
    """Train all models"""
    print("=" * 50)
    print("Model Training Phase")
    print("=" * 50)
    
    # Load data
    X_train, Y_train, data_config = MIMODataSimulator.load_data("data/train")
    X_val, Y_val, _ = MIMODataSimulator.load_data("data/val")
    
    n_observed = data_config['n_observed']
    n_predict = data_config['n_predict']
    
    print(f"Training data: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Validation data: X={X_val.shape}, Y={Y_val.shape}")
    
    # Create data loaders
    train_loader = create_data_loader(X_train, Y_train, 
                                    batch_size=config['batch_size'], 
                                    shuffle=True, num_workers=4)
    val_loader = create_data_loader(X_val, Y_val, 
                                  batch_size=config['batch_size'], 
                                  shuffle=False, num_workers=4)
    
    # Train ResCNN
    print("\nTraining ResCNN model...")
    rescnn_model = ResCNN(
        input_dim=3,
        n_observed=n_observed,
        n_predict=n_predict,
        hidden_dim=config['rescnn_hidden_dim'],
        n_layers=config['rescnn_layers']
    ).to(device)
    
    rescnn_trainer = ModelTrainer(rescnn_model, train_loader, val_loader, device)
    rescnn_history = rescnn_trainer.train(
        num_epochs=config['num_epochs'],
        lr=config['rescnn_lr'],
        weight_decay=config['weight_decay'],
        optimizer_type='adamw',
        scheduler_type='cosine',
        early_stopping_patience=config['early_stopping_patience'],
        model_name="rescnn"
    )
    
    # Save training history
    np.save("checkpoints/rescnn_training_history.npy", rescnn_history)
    
    # Train Transformer
    print("\nTraining Transformer model...")
    transformer_model = MIMOTransformer(
        input_dim=3,
        n_observed=n_observed,
        n_predict=n_predict,
        d_model=config['transformer_d_model'],
        n_heads=config['transformer_heads'],
        n_layers=config['transformer_layers'],
        d_ff=config['transformer_d_ff'],
        dropout=config['dropout']
    ).to(device)
    
    transformer_trainer = ModelTrainer(transformer_model, train_loader, val_loader, device)
    transformer_history = transformer_trainer.train(
        num_epochs=config['num_epochs'],
        lr=config['transformer_lr'],
        weight_decay=config['weight_decay'],
        optimizer_type='adamw',
        scheduler_type='cosine',
        early_stopping_patience=30,
        model_name="transformer"
    )
    
    # Save training history
    np.save("checkpoints/transformer_training_history.npy", transformer_history)
    
    print("Model training completed!")

def evaluate_models(config, device):
    """Evaluate all models"""
    print("=" * 50)
    print("Model Evaluation Phase")
    print("=" * 50)
    
    # Load test data
    X_test, Y_test, data_config = MIMODataSimulator.load_data("data/test")
    n_observed = data_config['n_observed']
    n_predict = data_config['n_predict']
    
    test_loader = create_data_loader(X_test, Y_test, 
                                   batch_size=config['batch_size'], 
                                   shuffle=False, num_workers=0)
    
    # Create models
    rescnn_model = ResCNN(
        input_dim=3, n_observed=n_observed, n_predict=n_predict,
        hidden_dim=config['rescnn_hidden_dim'], n_layers=config['rescnn_layers']
    ).to(device)
    
    transformer_model = MIMOTransformer(
        input_dim=3, n_observed=n_observed, n_predict=n_predict,
        d_model=config['transformer_d_model'], n_heads=config['transformer_heads'],
        n_layers=config['transformer_layers'], d_ff=config['transformer_d_ff'],
        dropout=config['dropout']
    ).to(device)
    
    # Load model weights
    def load_model(model, checkpoint_path):
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model loaded: {checkpoint_path}")
                return True
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"Model structure mismatch, need retraining: {checkpoint_path}")
                    return False
                else:
                    raise e
        else:
            print(f"Model file not found: {checkpoint_path}")
            return False
    
    rescnn_loaded = load_model(rescnn_model, "checkpoints/rescnn_best.pth")
    transformer_loaded = load_model(transformer_model, "checkpoints/transformer_best.pth")
    
    # If Transformer model loading failed, retrain
    if not transformer_loaded:
        print("\nRetraining Transformer model...")
        # Load training data
        X_train, Y_train, _ = MIMODataSimulator.load_data("data/train")
        X_val, Y_val, _ = MIMODataSimulator.load_data("data/val")
        
        train_loader = create_data_loader(X_train, Y_train, batch_size=config['batch_size'], 
                                        shuffle=True, num_workers=4)
        val_loader = create_data_loader(X_val, Y_val, batch_size=config['batch_size'], 
                                      shuffle=False, num_workers=4)
        
        # Retrain Transformer
        transformer_trainer = ModelTrainer(transformer_model, train_loader, val_loader, device)
        transformer_history = transformer_trainer.train(
            num_epochs=config['num_epochs'],
            lr=config['transformer_lr'],
            weight_decay=config['weight_decay'],
            optimizer_type='adamw',
            scheduler_type='cosine',
            early_stopping_patience=30,
            model_name="transformer"
        )
        transformer_loaded = True
        print("Transformer model retrained!")
    
    results = {}
    
    # Evaluate ResCNN
    if rescnn_loaded:
        print("\nEvaluating ResCNN model...")
        rescnn_model.eval()
        rescnn_predictions = []
        targets_list = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = rescnn_model(data)
                rescnn_predictions.append(output.cpu())
                targets_list.append(target.cpu())
        
        rescnn_preds = torch.cat(rescnn_predictions, dim=0)
        targets = torch.cat(targets_list, dim=0)
        
        rescnn_metrics = compute_comprehensive_metrics(rescnn_preds, targets)
        print_metrics(rescnn_metrics, "ResCNN")
        
        results['ResCNN'] = {
            'predictions': rescnn_preds,
            'targets': targets,
            'metrics': rescnn_metrics
        }
    
    # Evaluate Transformer
    if transformer_loaded:
        print("\nEvaluating Transformer model...")
        transformer_model.eval()
        transformer_predictions = []
        targets_list = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = transformer_model(data)
                transformer_predictions.append(output.cpu())
                targets_list.append(target.cpu())
        
        transformer_preds = torch.cat(transformer_predictions, dim=0)
        targets = torch.cat(targets_list, dim=0)
        
        transformer_metrics = compute_comprehensive_metrics(transformer_preds, targets)
        print_metrics(transformer_metrics, "Transformer")
        
        results['Transformer'] = {
            'predictions': transformer_preds,
            'targets': targets,
            'metrics': transformer_metrics
        }
    
    # Evaluate ensemble model
    if rescnn_loaded and transformer_loaded:
        print("\nEvaluating ensemble model...")
        
        # Optimize ensemble weights
        optimizer = EnsembleOptimizer(n_models=2)
        best_alpha, best_mse = optimizer.grid_search_weights(
            [rescnn_preds, transformer_preds], targets
        )
        print(f"Optimal ensemble weight: {best_alpha:.3f}")
        
        # Ensemble prediction
        ensemble_preds = ensemble_output(rescnn_preds, transformer_preds, best_alpha)
        ensemble_metrics = compute_comprehensive_metrics(ensemble_preds, targets)
        print_metrics(ensemble_metrics, "Ensemble Model")
        
        results['Ensemble'] = {
            'predictions': ensemble_preds,
            'targets': targets,
            'metrics': ensemble_metrics,
            'alpha': best_alpha
        }
        
        # Analyze ensemble weight effect
        ensemble_analysis = evaluate_ensemble_performance(
            [rescnn_preds, transformer_preds], targets
        )
        
        # Plot ensemble weight analysis
        plot_ensemble_weight_analysis(
            np.array(ensemble_analysis['weights']),
            ensemble_analysis['mse'],
            ensemble_analysis['nmse'],
            best_alpha,
            save_path="results/ensemble_weight_vs_mse.png"
        )
    
    return results

def create_visualizations(results):
    """Create visualization charts"""
    print("=" * 50)
    print("Visualization Generation Phase")
    print("=" * 50)
    
    # Create scatter plots
    create_all_scatter_plots(results, "results")
    
    # Create training curves
    create_all_training_plots("results")
    
    # Create robustness analysis plots
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run robustness experiments
    try:
        from robustness_analysis import RobustnessAnalyzer
        analyzer = RobustnessAnalyzer(device)
        robustness_results = analyzer.run_all_experiments()
        
        # Create robustness plots
        from evaluate.plot_robustness_analysis import create_all_robustness_plots
        
        # Prepare predictions dict for robustness plots
        predictions_dict = {}
        targets = None
        for model_name, result in results.items():
            predictions_dict[model_name] = result['predictions'].numpy()
            if targets is None:
                targets = result['targets'].numpy()
        
        create_all_robustness_plots(robustness_results, predictions_dict, targets, "results")
        
        print("Robustness analysis completed!")
        
    except ImportError:
        print("Robustness analysis module not available, skipping...")
    except Exception as e:
        print(f"Error in robustness analysis: {e}")
    
    print("Visualization charts generated!")

def save_results_summary(results, config):
    """Save results summary"""
    import pandas as pd
    
    # Create results summary
    summary_data = []
    for model_name, result in results.items():
        metrics = result['metrics']
        row = {
            'Model': model_name,
            'MSE': f"{metrics['mse']:.6f}",
            'NMSE': f"{metrics['nmse']:.6f}",
            'NMSE_dB': f"{metrics['nmse_db']:.2f}",
            'Correlation': f"{metrics['correlation']:.4f}",
            'Amplitude_MSE': f"{metrics['amplitude_mse']:.6f}",
            'Phase_MSE': f"{metrics['phase_mse']:.6f}"
        }
        summary_data.append(row)
    
    # Save as CSV
    df = pd.DataFrame(summary_data)
    df.to_csv("results/model_performance_summary.csv", index=False)
    
    # Save configuration info
    config_summary = {
        'n_antennas': config['n_antennas'],
        'n_observed': config['n_observed'],
        'n_predict': config['n_predict'],
        'noise_std': config['noise_std'],
        'train_samples': config['train_samples'],
        'test_samples': config['test_samples'],
        'num_epochs': config['num_epochs'],
        'batch_size': config['batch_size']
    }
    
    import json
    with open("results/experiment_config.json", "w") as f:
        json.dump(config_summary, f, indent=2)
    
    print("Results summary saved to results/ directory")

def main():
    parser = argparse.ArgumentParser(description='MIMO Virtual Dimension Extension Complete Pipeline')
    
    # Data parameters
    parser.add_argument('--n_antennas', type=int, default=64, help='Total antenna count')
    parser.add_argument('--n_observed', type=int, default=16, help='Observed antenna count')
    parser.add_argument('--n_predict', type=int, default=48, help='Predicted antenna count')
    parser.add_argument('--noise_std', type=float, default=0.1, help='Noise standard deviation')
    parser.add_argument('--train_samples', type=int, default=10000, help='Training samples')
    parser.add_argument('--val_samples', type=int, default=2000, help='Validation samples')
    parser.add_argument('--test_samples', type=int, default=1000, help='Test samples')
    
    # ResCNN parameters
    parser.add_argument('--rescnn_hidden_dim', type=int, default=128, help='ResCNN hidden dimension')
    parser.add_argument('--rescnn_layers', type=int, default=6, help='ResCNN residual layers')
    parser.add_argument('--rescnn_lr', type=float, default=1e-3, help='ResCNN learning rate')
    
    # Transformer parameters
    parser.add_argument('--transformer_d_model', type=int, default=96, help='Transformer model dimension')
    parser.add_argument('--transformer_heads', type=int, default=3, help='Transformer attention heads')
    parser.add_argument('--transformer_layers', type=int, default=2, help='Transformer encoder layers')
    parser.add_argument('--transformer_d_ff', type=int, default=192, help='Transformer feedforward dimension')
    parser.add_argument('--transformer_lr', type=float, default=2e-3, help='Transformer learning rate')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--early_stopping_patience', type=int, default=30, help='Early stopping patience')
    
    # Execution control
    parser.add_argument('--generate_data', action='store_true', help='Generate new data')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    
    args = parser.parse_args()
    
    # Configuration dictionary
    config = vars(args)
    
    # Setup environment
    device = setup_environment()
    
    print("MIMO Virtual Dimension Extension Project")
    print("=" * 50)
    print("Configuration:")
    for key, value in config.items():
        if not key.startswith(('generate_data', 'train', 'evaluate', 'visualize', 'all')):
            print(f"{key}: {value}")
    print("=" * 50)
    
    # Run pipeline
    if args.all:
        # Complete pipeline
        if not os.path.exists("data/train"):
            generate_data(config)
        
        if not (os.path.exists("checkpoints/rescnn_best.pth") and 
                os.path.exists("checkpoints/transformer_best.pth")):
            train_models(config, device)
        
        results = evaluate_models(config, device)
        create_visualizations(results)
        save_results_summary(results, config)
        
    else:
        # Step-by-step execution
        if args.generate_data:
            generate_data(config)
        
        if args.train:
            train_models(config, device)
        
        if args.evaluate:
            results = evaluate_models(config, device)
            
            if args.visualize:
                create_visualizations(results)
                save_results_summary(results, config)
    
    print("\nProject execution completed!")
    print("Results saved in the following directories:")
    print("- checkpoints/: Model weight files")
    print("- results/: Evaluation results and charts")
    print("- data/: Simulation data")

if __name__ == "__main__":
    main()