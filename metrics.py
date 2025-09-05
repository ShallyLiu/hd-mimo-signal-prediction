"""
评估指标计算模块
"""

import torch
import numpy as np
from typing import Tuple, Dict, Union, List

def compute_mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    计算均方误差 (MSE)
    
    Args:
        predictions: 预测值 [batch_size, n_predict, 2]
        targets: 真实值 [batch_size, n_predict, 2]
        
    Returns:
        MSE值
    """
    mse = torch.mean((predictions - targets) ** 2)
    return mse.item()

def compute_nmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    计算归一化均方误差 (NMSE)
    
    Args:
        predictions: 预测值 [batch_size, n_predict, 2]
        targets: 真实值 [batch_size, n_predict, 2]
        
    Returns:
        NMSE值
    """
    # 转换为复数
    pred_complex = predictions[:, :, 0] + 1j * predictions[:, :, 1]
    target_complex = targets[:, :, 0] + 1j * targets[:, :, 1]
    
    # 计算NMSE
    numerator = torch.mean(torch.abs(pred_complex - target_complex) ** 2)
    denominator = torch.mean(torch.abs(target_complex) ** 2)
    
    nmse = numerator / (denominator + 1e-8)
    return nmse.item()

def compute_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    计算平均绝对误差 (MAE)
    
    Args:
        predictions: 预测值
        targets: 真实值
        
    Returns:
        MAE值
    """
    mae = torch.mean(torch.abs(predictions - targets))
    return mae.item()

def compute_complex_mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    计算复数信号的MSE
    
    Args:
        predictions: 预测值 [batch_size, n_predict, 2]
        targets: 真实值 [batch_size, n_predict, 2]
        
    Returns:
        复数MSE值
    """
    # 转换为复数
    pred_complex = predictions[:, :, 0] + 1j * predictions[:, :, 1]
    target_complex = targets[:, :, 0] + 1j * targets[:, :, 1]
    
    # 计算复数MSE
    complex_mse = torch.mean(torch.abs(pred_complex - target_complex) ** 2)
    return complex_mse.item()

def compute_correlation_coefficient(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    计算相关系数
    
    Args:
        predictions: 预测值
        targets: 真实值
        
    Returns:
        相关系数
    """
    # 展平张量
    pred_flat = predictions.view(-1)
    target_flat = targets.view(-1)
    
    # 计算相关系数
    vx = pred_flat - torch.mean(pred_flat)
    vy = target_flat - torch.mean(target_flat)
    
    correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return correlation.item()

def compute_snr_improvement(predictions: torch.Tensor, 
                          targets: torch.Tensor,
                          noise_power: float = 0.01) -> float:
    """
    计算信噪比改善
    
    Args:
        predictions: 预测值
        targets: 真实值
        noise_power: 噪声功率
        
    Returns:
        SNR改善 (dB)
    """
    # 计算信号功率
    signal_power = torch.mean(targets ** 2)
    
    # 计算误差功率
    error_power = torch.mean((predictions - targets) ** 2)
    
    # 计算SNR
    input_snr = 10 * torch.log10(signal_power / noise_power + 1e-8)
    output_snr = 10 * torch.log10(signal_power / error_power + 1e-8)
    
    snr_improvement = output_snr - input_snr
    return snr_improvement.item()

def compute_amplitude_error(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    计算幅度误差指标
    
    Args:
        predictions: 预测值 [batch_size, n_predict, 2]
        targets: 真实值 [batch_size, n_predict, 2]
        
    Returns:
        幅度误差字典
    """
    # 转换为复数并计算幅度
    pred_complex = predictions[:, :, 0] + 1j * predictions[:, :, 1]
    target_complex = targets[:, :, 0] + 1j * targets[:, :, 1]
    
    pred_amplitude = torch.abs(pred_complex)
    target_amplitude = torch.abs(target_complex)
    
    # 计算各种幅度误差
    amplitude_mse = torch.mean((pred_amplitude - target_amplitude) ** 2).item()
    amplitude_mae = torch.mean(torch.abs(pred_amplitude - target_amplitude)).item()
    amplitude_nmse = (torch.mean((pred_amplitude - target_amplitude) ** 2) / 
                     torch.mean(target_amplitude ** 2)).item()
    
    return {
        'amplitude_mse': amplitude_mse,
        'amplitude_mae': amplitude_mae,
        'amplitude_nmse': amplitude_nmse
    }

def compute_phase_error(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    计算相位误差指标
    
    Args:
        predictions: 预测值 [batch_size, n_predict, 2]
        targets: 真实值 [batch_size, n_predict, 2]
        
    Returns:
        相位误差字典
    """
    # 转换为复数并计算相位
    pred_complex = predictions[:, :, 0] + 1j * predictions[:, :, 1]
    target_complex = targets[:, :, 0] + 1j * targets[:, :, 1]
    
    pred_phase = torch.angle(pred_complex)
    target_phase = torch.angle(target_complex)
    
    # 处理相位差的周期性
    phase_diff = pred_phase - target_phase
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    
    # 计算相位误差
    phase_mse = torch.mean(phase_diff ** 2).item()
    phase_mae = torch.mean(torch.abs(phase_diff)).item()
    phase_std = torch.std(phase_diff).item()
    
    return {
        'phase_mse': phase_mse,
        'phase_mae': phase_mae,
        'phase_std': phase_std
    }

def compute_comprehensive_metrics(predictions: torch.Tensor, 
                                targets: torch.Tensor) -> Dict[str, float]:
    """
    计算综合评估指标
    
    Args:
        predictions: 预测值 [batch_size, n_predict, 2]
        targets: 真实值 [batch_size, n_predict, 2]
        
    Returns:
        综合指标字典
    """
    metrics = {}
    
    # 基本指标
    metrics['mse'] = compute_mse(predictions, targets)
    metrics['nmse'] = compute_nmse(predictions, targets)
    metrics['mae'] = compute_mae(predictions, targets)
    metrics['complex_mse'] = compute_complex_mse(predictions, targets)
    metrics['correlation'] = compute_correlation_coefficient(predictions, targets)
    
    # 幅度误差
    amplitude_metrics = compute_amplitude_error(predictions, targets)
    metrics.update(amplitude_metrics)
    
    # 相位误差
    phase_metrics = compute_phase_error(predictions, targets)
    metrics.update(phase_metrics)
    
    # 转换NMSE到dB
    metrics['nmse_db'] = 10 * np.log10(metrics['nmse'] + 1e-8)
    
    return metrics

def print_metrics(metrics: Dict[str, float], model_name: str = "模型"):
    """
    打印评估指标
    
    Args:
        metrics: 指标字典
        model_name: 模型名称
    """
    print(f"\n{model_name} 性能指标:")
    print("-" * 40)
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"NMSE: {metrics['nmse']:.6f} ({metrics['nmse_db']:.2f} dB)")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"复数MSE: {metrics['complex_mse']:.6f}")
    print(f"相关系数: {metrics['correlation']:.4f}")
    
    print(f"\n幅度误差:")
    print(f"  MSE: {metrics['amplitude_mse']:.6f}")
    print(f"  MAE: {metrics['amplitude_mae']:.6f}")
    print(f"  NMSE: {metrics['amplitude_nmse']:.6f}")
    
    print(f"\n相位误差:")
    print(f"  MSE: {metrics['phase_mse']:.6f}")
    print(f"  MAE: {metrics['phase_mae']:.6f}")
    print(f"  标准差: {metrics['phase_std']:.6f}")

def compare_models(metrics_list: List[Dict[str, float]], 
                  model_names: List[str]) -> Dict[str, str]:
    """
    比较多个模型的性能
    
    Args:
        metrics_list: 指标列表
        model_names: 模型名称列表
        
    Returns:
        最佳模型字典
    """
    comparison = {}
    
    # 比较主要指标 (越小越好)
    for metric in ['mse', 'nmse', 'mae', 'complex_mse']:
        best_idx = np.argmin([m[metric] for m in metrics_list])
        comparison[f'best_{metric}'] = model_names[best_idx]
    
    # 比较相关系数 (越大越好)
    best_idx = np.argmax([m['correlation'] for m in metrics_list])
    comparison['best_correlation'] = model_names[best_idx]
    
    return comparison

if __name__ == "__main__":
    # 测试指标计算
    batch_size, n_predict = 100, 48
    
    # 生成测试数据
    targets = torch.randn(batch_size, n_predict, 2)
    predictions = targets + 0.1 * torch.randn_like(targets)  # 添加少量噪声
    
    # 计算指标
    metrics = compute_comprehensive_metrics(predictions, targets)
    print_metrics(metrics, "测试模型")
    
    print("\n指标计算模块测试完成")