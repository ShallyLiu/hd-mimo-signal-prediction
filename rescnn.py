"""
ResCNN模型实现
基于残差卷积神经网络的MIMO信号预测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 如果输入输出通道数不同，需要调整shortcut连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = F.relu(out)
        
        return out

class ResCNN(nn.Module):
    """
    ResCNN模型
    用于MIMO信号虚拟维度扩展
    """
    
    def __init__(self, 
                 input_dim: int = 3,
                 n_observed: int = 16,
                 n_predict: int = 48,
                 hidden_dim: int = 128,
                 n_layers: int = 6):
        """
        初始化ResCNN模型
        
        Args:
            input_dim: 输入特征维度 (Re/Im/位置)
            n_observed: 观测天线数量
            n_predict: 预测天线数量
            hidden_dim: 隐藏层维度
            n_layers: 残差块层数
        """
        super(ResCNN, self).__init__()
        
        self.input_dim = input_dim
        self.n_observed = n_observed
        self.n_predict = n_predict
        self.hidden_dim = hidden_dim
        
        # 输入嵌入层
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, n_observed, hidden_dim) * 0.1)
        
        # 将输入转换为卷积格式
        self.conv_input = nn.Conv1d(hidden_dim, hidden_dim, 1)
        
        # 残差块堆叠
        self.res_blocks = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.res_blocks.append(ResidualBlock(hidden_dim, hidden_dim))
            else:
                self.res_blocks.append(ResidualBlock(hidden_dim, hidden_dim))
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1, batch_first=True)
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_predict * 2)  # 输出预测位置的Re和Im
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, n_observed, input_dim]
            
        Returns:
            输出张量 [batch_size, n_predict, 2]
        """
        batch_size = x.size(0)
        
        # 输入嵌入
        x = self.input_embedding(x)  # [batch_size, n_observed, hidden_dim]
        
        # 添加位置编码
        x = x + self.pos_encoding
        
        # 残差连接的输入
        residual_input = x
        
        # 转换为卷积格式 [batch_size, hidden_dim, n_observed]
        x = x.transpose(1, 2)
        x = self.conv_input(x)
        
        # 通过残差块
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 转换回序列格式 [batch_size, n_observed, hidden_dim]
        x = x.transpose(1, 2)
        
        # 残差连接
        x = x + residual_input
        
        # 自注意力机制
        x_attn, _ = self.attention(x, x, x)
        x = x + x_attn
        
        # 全局平均池化
        x = torch.mean(x, dim=1)  # [batch_size, hidden_dim]
        
        # 输出投影
        output = self.output_projection(x)  # [batch_size, n_predict * 2]
        
        # 重塑为 [batch_size, n_predict, 2]
        output = output.view(batch_size, self.n_predict, 2)
        
        return output
    
    def get_model_size(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = ResCNN(
        input_dim=3,
        n_observed=16,
        n_predict=48,
        hidden_dim=128,
        n_layers=6
    ).to(device)
    
    print(f"模型参数数量: {model.get_model_size():,}")
    print(f"可训练参数数量: {model.get_trainable_params():,}")
    
    # 测试前向传播
    batch_size = 32
    x = torch.randn(batch_size, 16, 3).to(device)
    
    with torch.no_grad():
        output = model(x)
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        
    print("ResCNN模型测试完成")