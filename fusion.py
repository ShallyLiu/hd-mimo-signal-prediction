"""
模型融合模块
提供多种模型融合策略
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional

def ensemble_output(output1: torch.Tensor, 
                   output2: torch.Tensor, 
                   alpha: float = 0.5) -> torch.Tensor:
    """
    简单加权融合两个模型的输出
    
    Args:
        output1: 第一个模型的输出 [batch_size, n_predict, 2]
        output2: 第二个模型的输出 [batch_size, n_predict, 2]
        alpha: 融合权重，alpha * output1 + (1-alpha) * output2
        
    Returns:
        融合后的输出
    """
    return alpha * output1 + (1 - alpha) * output2

def weighted_ensemble(outputs: List[torch.Tensor], 
                     weights: List[float]) -> torch.Tensor:
    """
    多模型加权融合
    
    Args:
        outputs: 模型输出列表
        weights: 权重列表
        
    Returns:
        融合后的输出
    """
    assert len(outputs) == len(weights), "输出和权重数量必须一致"
    assert abs(sum(weights) - 1.0) < 1e-6, "权重和必须为1"
    
    ensemble_output = torch.zeros_like(outputs[0])
    for output, weight in zip(outputs, weights):
        ensemble_output += weight * output
        
    return ensemble_output

class AdaptiveEnsemble(nn.Module):
    """自适应融合模块"""
    
    def __init__(self, 
                 input_dim: int,
                 n_models: int = 2,
                 hidden_dim: int = 64):
        """
        初始化自适应融合模块
        
        Args:
            input_dim: 输入特征维度
            n_models: 模型数量
            hidden_dim: 隐藏层维度
        """
        super(AdaptiveEnsemble, self).__init__()
        
        self.n_models = n_models
        
        # 权重预测网络
        self.weight_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_models),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, 
                input_features: torch.Tensor,
                model_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_features: 输入特征 [batch_size, input_dim]
            model_outputs: 模型输出列表
            
        Returns:
            融合输出, 权重
        """
        # 预测权重
        weights = self.weight_predictor(input_features)  # [batch_size, n_models]
        
        # 加权融合
        ensemble_output = torch.zeros_like(model_outputs[0])
        for i, output in enumerate(model_outputs):
            weight = weights[:, i:i+1].unsqueeze(-1)  # [batch_size, 1, 1]
            ensemble_output += weight * output
            
        return ensemble_output, weights

class VarianceBasedEnsemble:
    """基于方差的动态融合"""
    
    @staticmethod
    def compute_prediction_variance(predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        计算预测方差
        
        Args:
            predictions: 模型预测列表
            
        Returns:
            预测方差
        """
        stacked_preds = torch.stack(predictions, dim=0)  # [n_models, batch_size, n_predict, 2]
        variance = torch.var(stacked_preds, dim=0)  # [batch_size, n_predict, 2]
        return torch.mean(variance, dim=-1)  # [batch_size, n_predict]
    
    @staticmethod
    def uncertainty_weighted_ensemble(predictions: List[torch.Tensor],
                                    uncertainties: List[torch.Tensor]) -> torch.Tensor:
        """
        基于不确定性的加权融合
        
        Args:
            predictions: 模型预测列表
            uncertainties: 不确定性列表
            
        Returns:
            融合预测
        """
        # 将不确定性转换为权重 (不确定性越小，权重越大)
        weights = []
        for uncertainty in uncertainties:
            weight = 1.0 / (uncertainty + 1e-8)
            weights.append(weight)
        
        # 归一化权重
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # 加权融合
        ensemble_pred = torch.zeros_like(predictions[0])
        for pred, weight in zip(predictions, normalized_weights):
            ensemble_pred += weight.unsqueeze(-1) * pred
            
        return ensemble_pred

class EnsembleOptimizer:
    """融合权重优化器"""
    
    def __init__(self, n_models: int = 2):
        self.n_models = n_models
        
    def grid_search_weights(self,
                          model_predictions: List[torch.Tensor],
                          targets: torch.Tensor,
                          alpha_range: np.ndarray = None) -> Tuple[float, float]:
        """
        网格搜索最优融合权重
        
        Args:
            model_predictions: 模型预测列表
            targets: 真实标签
            alpha_range: 搜索范围
            
        Returns:
            最优权重, 最优MSE
        """
        if alpha_range is None:
            alpha_range = np.linspace(0, 1, 21)
            
        best_alpha = 0.5
        best_mse = float('inf')
        
        for alpha in alpha_range:
            # 融合预测
            ensemble_pred = ensemble_output(model_predictions[0], model_predictions[1], alpha)
            
            # 计算MSE
            mse = torch.mean((ensemble_pred - targets) ** 2).item()
            
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
                
        return best_alpha, best_mse
    
    def bayesian_optimization_weights(self,
                                    model_predictions: List[torch.Tensor],
                                    targets: torch.Tensor,
                                    n_iterations: int = 50) -> Tuple[List[float], float]:
        """
        贝叶斯优化搜索最优权重
        
        Args:
            model_predictions: 模型预测列表
            targets: 真实标签
            n_iterations: 迭代次数
            
        Returns:
            最优权重列表, 最优MSE
        """
        from scipy.optimize import minimize
        
        def objective(weights):
            # 归一化权重
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # 融合预测
            ensemble_pred = torch.zeros_like(model_predictions[0])
            for i, pred in enumerate(model_predictions):
                ensemble_pred += weights[i] * pred
                
            # 计算MSE
            mse = torch.mean((ensemble_pred - targets) ** 2).item()
            return mse
        
        # 初始权重
        initial_weights = [1.0 / self.n_models] * self.n_models
        
        # 约束条件：权重和为1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = [(0, 1)] * self.n_models
        
        # 优化
        result = minimize(objective, initial_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x.tolist(), result.fun

def evaluate_ensemble_performance(predictions: List[torch.Tensor],
                                targets: torch.Tensor,
                                weights_range: np.ndarray = None) -> Dict[str, float]:
    """
    评估不同融合权重下的性能
    
    Args:
        predictions: 模型预测列表
        targets: 真实标签
        weights_range: 权重搜索范围
        
    Returns:
        性能字典
    """
    if weights_range is None:
        weights_range = np.linspace(0, 1, 21)
    
    results = {
        'weights': [],
        'mse': [],
        'nmse': []
    }
    
    for alpha in weights_range:
        # 融合预测
        ensemble_pred = ensemble_output(predictions[0], predictions[1], alpha)
        
        # 计算指标
        mse = torch.mean((ensemble_pred - targets) ** 2).item()
        
        # 计算NMSE
        pred_complex = ensemble_pred[:, :, 0] + 1j * ensemble_pred[:, :, 1]
        target_complex = targets[:, :, 0] + 1j * targets[:, :, 1]
        nmse = (torch.mean(torch.abs(pred_complex - target_complex) ** 2) / 
                torch.mean(torch.abs(target_complex) ** 2)).item()
        
        results['weights'].append(alpha)
        results['mse'].append(mse)
        results['nmse'].append(nmse)
    
    return results

if __name__ == "__main__":
    # 测试融合模块
    batch_size, n_predict = 32, 48
    
    # 模拟两个模型的输出
    output1 = torch.randn(batch_size, n_predict, 2)
    output2 = torch.randn(batch_size, n_predict, 2)
    targets = torch.randn(batch_size, n_predict, 2)
    
    # 测试简单融合
    ensemble_pred = ensemble_output(output1, output2, alpha=0.6)
    print(f"融合输出形状: {ensemble_pred.shape}")
    
    # 测试权重优化
    optimizer = EnsembleOptimizer(n_models=2)
    best_alpha, best_mse = optimizer.grid_search_weights([output1, output2], targets)
    print(f"最优权重: {best_alpha:.3f}, 最优MSE: {best_mse:.6f}")
    
    # 测试自适应融合
    input_features = torch.randn(batch_size, 16)  # 假设输入特征维度为16
    adaptive_ensemble = AdaptiveEnsemble(input_dim=16, n_models=2)
    
    with torch.no_grad():
        adaptive_pred, weights = adaptive_ensemble(input_features, [output1, output2])
        print(f"自适应融合输出形状: {adaptive_pred.shape}")
        print(f"预测权重: {weights[0]}")
    
    print("融合模块测试完成")