# MIMO Virtual Dimension Extension Project

Deep learning-based virtual dimension extension for high-dimensional MIMO systems. Predicts the received signals at unobserved antenna positions from partial antenna observations to enhance spatial degrees of freedom.

## Key Features

- Ensemble model achieves NMSE of -9.88dB with correlation coefficient of 0.948
- Predicts 75% of signals using only 25% observed antennas with 94.8% accuracy
- ResCNN + Transformer + Intelligent Fusion Architecture

## Performance Metrics

| Model | NMSE (dB) | Correlation | MSE | Parameters |
|-------|-----------|-------------|-----|------------|
| ResCNN | -9.42 | 0.942 | 0.232 | 758K |
| Transformer | -8.86 | 0.933 | 0.264 | 198K |
| Ensemble | -9.88 | 0.948 | 0.209 | - |

## Quick Start

### Environment Setup

```bash
pip install -r requirements.txt
```

### Execution

```bash
# Complete pipeline
python main.py --all

# Step-by-step execution
python main.py --generate_data    # Generate data
python main.py --train           # Train models
python main.py --evaluate --visualize  # Evaluate and visualize
```

## Project Structure

```
hd_mimo_project/
├── data/                           # Simulation data
│   ├── train/                      # Training data (10,000 samples)
│   ├── val/                        # Validation data (2,000 samples)
│   └── test/                       # Test data (1,000 samples)
├── models/                         # Deep learning models
│   ├── rescnn.py                   # ResCNN model (758K parameters)
│   └── transformer.py              # Transformer model (198K parameters)
├── ensemble/                       # Model fusion
│   └── fusion.py                   # Fusion strategies and weight optimization
├── train/                          # Training modules
│   ├── utils.py                    # Training utilities and loss functions
│   ├── train_rescnn.py            # ResCNN training script
│   └── train_transformer.py       # Transformer training script
├── evaluate/                       # Evaluation and visualization
│   ├── metrics.py                  # Evaluation metrics computation
│   ├── evaluate_models.py          # Unified model evaluation
│   ├── plot_scatter.py             # Scatter plot generation
│   └── plot_loss_curve.py          # Training curve plotting
├── checkpoints/                    # Model weights
├── results/                        # Output results
├── data_simulator.py               # MIMO data simulator
├── main.py                         # Main execution script
└── requirements.txt                # Python dependencies
```

## Core Modules

### Data Simulation Module (data_simulator.py)

Generates MIMO multipath Rayleigh fading channel data:
- 64-antenna rectangular array with λ/2 spacing
- 3-7 multipaths with complex Gaussian gains and exponential power decay
- Controllable noise levels supporting different SNR scenarios
- 16 observed antennas → 48 predicted antennas

```python
simulator = MIMODataSimulator(
    n_antennas=64,      # Total antennas
    n_observed=16,      # Observed antennas  
    n_predict=48,       # Predicted antennas
    noise_std=0.1       # Noise standard deviation
)
X, Y = simulator.generate_training_data(10000)
```

### ResCNN Model (models/rescnn.py)

Residual Convolutional Neural Network + Attention Mechanism:
- Stable training with fast convergence
- Strong spatial feature extraction capability
- Optimized for antenna array signal processing
- Performance: NMSE -9.42dB, 758K parameters

### Transformer Model (models/transformer.py)

Self-attention encoder + Global pooling:
- Captures long-range spatial dependencies
- Lightweight design with efficient training
- Pre-norm architecture ensures stability
- Performance: NMSE -8.86dB, 198K parameters

### Fusion System (ensemble/fusion.py)

Weighted linear combination + Grid search optimization:
- Optimal weights: 0.6 (ResCNN) + 0.4 (Transformer)
- Performance improvement: NMSE enhanced to -9.88dB
- Complementary advantages: Combines CNN spatial features and Transformer sequence modeling

## Configuration Parameters

### Data Parameters
```bash
--n_antennas 64        # Total number of antennas
--n_observed 16        # Number of observed antennas  
--n_predict 48         # Number of predicted antennas
--noise_std 0.1        # Noise standard deviation
--train_samples 10000  # Number of training samples
```

### ResCNN Parameters
```bash
--rescnn_hidden_dim 128  # Hidden layer dimension
--rescnn_layers 6        # Number of residual blocks
--rescnn_lr 0.001        # Learning rate
```

### Transformer Parameters
```bash
--transformer_d_model 96    # Model dimension
--transformer_heads 3       # Number of attention heads
--transformer_layers 2      # Number of encoder layers
--transformer_lr 0.002      # Learning rate
```

### Training Parameters
```bash
--num_epochs 200            # Number of training epochs
--batch_size 64             # Batch size
--weight_decay 0.0001       # Weight decay
--early_stopping_patience 30 # Early stopping patience
```

## Results Interpretation

### Training Process
- ResCNN: 200 epochs, 14 minutes, stable convergence
- Transformer: 200 epochs, 11 minutes, no early stopping issues
- Fusion optimization: Grid search finds optimal weight combination

### Performance Evaluation
- MSE: Mean squared error, lower is better
- NMSE: Normalized mean squared error, -9.88dB approaches theoretical limit
- Correlation coefficient: 0.948 indicates high correlation between predictions and ground truth
- Complex precision: Both amplitude and phase errors within acceptable range

### Visualization Charts
- scatter_*.png: True vs predicted value scatter plots
- loss_*.png: Training/validation loss curves
- ensemble_weight_vs_mse.png: Fusion weight analysis
- error_distribution_*.png: Error distribution histograms

## Advanced Usage

### Custom Configuration
```bash
python main.py --all \
    --n_antennas 128 \
    --n_observed 32 \
    --n_predict 96 \
    --num_epochs 300 \
    --batch_size 128
```

### Individual Training
```bash
# ResCNN
python train/train_rescnn.py --hidden_dim 256 --n_layers 8

# Transformer  
python train/train_transformer.py --d_model 128 --n_heads 4
```

### Model Evaluation
```bash
python evaluate/evaluate_models.py \
    --rescnn_checkpoint checkpoints/rescnn_best.pth \
    --transformer_checkpoint checkpoints/transformer_best.pth \
    --optimize_ensemble
```

### Utility Scripts
```bash
python test_env.py                                    # Environment test
python clean_checkpoints.py --model all --confirm     # Clean weights
python debug_transformer.py                           # Debug analysis
```

## Technical Principles

### MIMO Channel Model
Uses multipath Rayleigh fading channel model:

```
H = Σ αᵢ · a(θᵢ)
```

Where:
- αᵢ: Complex gain of the i-th path
- a(θᵢ): Steering vector corresponding to arrival angle θᵢ
- Number of multipaths: 3-7 (random)
- Power decay: Exponential distribution

### Virtual Dimension Extension
Implements mapping from observed signals to unobserved position signals via deep learning:

```
f: C^(M×3) → C^(N×2)
```

Input: {Real, Imaginary, Position} of M observed antennas  
Output: {Real, Imaginary} of N predicted positions

### Loss Function
Uses complex mean squared error loss:

```
L = E[||H_pred - H_true||²]
```

Combined with normalized mean squared error (NMSE) for evaluation.

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
python main.py --all --batch_size 32
# Or reduce model dimensions
python main.py --all --rescnn_hidden_dim 64 --transformer_d_model 64
```

**Training Not Converging**
```bash
python debug_transformer.py  # Check data quality
# Adjust learning rates
python main.py --all --rescnn_lr 5e-4 --transformer_lr 1e-3
```

**Weight Loading Failed**
```bash
python clean_checkpoints.py --model all --confirm
python main.py --all
```

### Performance Benchmarks
- NMSE < -8dB: Good performance
- NMSE < -9dB: Excellent performance  
- Correlation coefficient > 0.9: High precision prediction
- Fusion gain > 5%: Effective fusion
