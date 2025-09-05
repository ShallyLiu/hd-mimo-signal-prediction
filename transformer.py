"""
Transformer model implementation
Transformer architecture optimized for MIMO signal prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MIMOTransformer(nn.Module):
    """
    Optimized Transformer model specifically designed for MIMO signal prediction
    """
    
    def __init__(self,
                 input_dim: int = 3,
                 n_observed: int = 16,
                 n_predict: int = 48,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 4,
                 d_ff: int = 256,
                 dropout: float = 0.1):
        """
        Initialize optimized Transformer model
        """
        super(MIMOTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.n_observed = n_observed
        self.n_predict = n_predict
        self.d_model = d_model
        
        # Improved input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Sinusoidal positional encoding
        pe = torch.zeros(n_observed, d_model)
        position = torch.arange(0, n_observed, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Learnable position embedding
        self.position_embedding = nn.Parameter(
            torch.randn(1, n_observed, d_model) * 0.01
        )
        
        # Multi-scale feature extraction convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(d_model, d_model // 4, kernel_size=k, padding=k//2)
            for k in [1, 3, 5, 7]
        ])
        self.conv_bn = nn.BatchNorm1d(d_model)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.transformer_layers.append(layer)
        
        # Attention pooling weights
        self.attention_weights = nn.Linear(d_model, 1)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),  # Fusion of 3 pooling methods
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(d_model * 2, d_model * 4),
            nn.LayerNorm(d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(d_model * 4, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(d_model * 2, n_predict * 2)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Improved weight initialization"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'output_head' in name and 'weight' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation
        
        Args:
            x: Input tensor [batch_size, n_observed, input_dim]
            
        Returns:
            Output tensor [batch_size, n_predict, 2]
        """
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_projection(x)  # [batch_size, n_observed, d_model]
        
        # Add positional encoding
        x = x + self.pe + self.position_embedding
        
        # Through transformer layers, apply multi-scale feature extraction after each layer
        for i, transformer_layer in enumerate(self.transformer_layers):
            # Transformer encoding
            x = transformer_layer(x)
            
            # Multi-scale feature extraction (applied in intermediate layers)
            if i < len(self.transformer_layers) - 1:
                residual = x
                x_conv = x.transpose(1, 2)  # [batch_size, d_model, n_observed]
                
                # Multi-scale convolution
                conv_outputs = []
                for conv in self.conv_layers:
                    conv_out = F.relu(conv(x_conv))
                    conv_outputs.append(conv_out)
                
                # Concatenate multi-scale features
                multi_scale = torch.cat(conv_outputs, dim=1)
                multi_scale = self.conv_bn(multi_scale)
                multi_scale = F.dropout(multi_scale, p=0.1, training=self.training)
                
                x = residual + multi_scale.transpose(1, 2)
        
        # Multiple pooling methods
        # 1. Attention weighted average
        attention_scores = F.softmax(self.attention_weights(x), dim=1)
        weighted_avg = torch.sum(x * attention_scores, dim=1)
        
        # 2. Global average pooling
        avg_pooled = torch.mean(x, dim=1)
        
        # 3. Global max pooling
        max_pooled, _ = torch.max(x, dim=1)
        
        # Fuse three pooling results
        pooled = torch.cat([weighted_avg, avg_pooled, max_pooled], dim=1)
        
        # Output projection
        output = self.output_head(pooled)  # [batch_size, n_predict * 2]
        
        # Reshape to [batch_size, n_predict, 2]
        output = output.view(batch_size, self.n_predict, 2)
        
        return output
    
    def get_model_size(self) -> int:
        """Get number of model parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MIMOTransformer(
        input_dim=3,
        n_observed=16,
        n_predict=48,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        dropout=0.1
    ).to(device)
    
    print(f"Number of model parameters: {model.get_model_size():,}")
    print(f"Number of trainable parameters: {model.get_trainable_params():,}")
    
    # Test forward propagation
    batch_size = 32
    x = torch.randn(batch_size, 16, 3).to(device)
    
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
    
    print("Transformer model test completed")