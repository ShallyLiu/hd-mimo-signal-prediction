"""
模型评估脚本
统一评估ResCNN、Transformer和融合模型的性能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import argparse

from models.rescnn import ResCNN
from models.transformer import MIMOTransformer
from ensemble.fusion import ensemble_output, EnsembleOptimizer
from evaluate.metrics import compute_comprehensive_metrics, print_metrics, compare_models
from data_simulator import MIMODataSimulator
from train.utils import create_data_loader

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.results = {}
        
    def load_model_checkpoint(self, 
                            model: torch.nn.Module, 
                            checkpoint_path: str) -> torch.nn.Module:
        """加载模型检查点"""
        if os.path.exists(checkpoint_path):
            print(f"加载模型: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            return model
        else:
            raise FileNotFoundError(f"未找到模型文件: {checkpoint_path}")
    
    def evaluate_single_model(self,
                            model: torch.nn.Module,
                            test_loader: torch.utils.data.DataLoader,
                            model_name: str) -> Dict[str, float]:
        """评估单个模型"""
        print(f"\n评估 {model_name} 模型...")
        
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                output = model(data)
                
                all_predictions.append(output.cpu())
                all_targets.append(target.cpu())
                
                if batch_idx % 10 == 0:
                    print(f"处理批次: {batch_idx+1}/{len(test_loader)}")
        
        # 合并所有批次
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # 计算指标
        metrics = compute_comprehensive_metrics(predictions, targets)
        print_metrics(metrics, model_name)
        
        self.results[model_name] = {
            'predictions': predictions,
            'targets': targets,
            'metrics': metrics
        }
        
        return metrics
    
    def evaluate_ensemble_model(self,
                              rescnn_model: torch.nn.Module,
                              transformer_model: torch.nn.Module,
                              test_loader: torch.utils.data.DataLoader,
                              optimize_weights: bool = True) -> Dict[str, float]:
        """评估融合模型"""
        print(f"\n评估融合模型...")
        
        rescnn_model.eval()
        transformer_model.eval()
        
        rescnn_predictions = []
        transformer_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # 两个模型的预测
                rescnn_output = rescnn_model(data)
                transformer_output = transformer_model(data)
                
                rescnn_predictions.append(rescnn_output.cpu())
                transformer_predictions.append(transformer_output.cpu())
                all_targets.append(target.cpu())
                
                if batch_idx % 10 == 0:
                    print(f"处理批次: {batch_idx+1}/{len(test_loader)}")
        
        # 合并预测
        rescnn_preds = torch.cat(rescnn_predictions, dim=0)
        transformer_preds = torch.cat(transformer_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # 优化融合权重
        if optimize_weights:
            optimizer = EnsembleOptimizer(n_models=2)
            best_alpha, best_mse = optimizer.grid_search_weights(
                [rescnn_preds, transformer_preds], targets
            )
            print(f"优化后的融合权重: {best_alpha:.3f} (ResCNN) / {1-best_alpha:.3f} (Transformer)")
        else:
            best_alpha = 0.5
        
        # 融合预测
        ensemble_preds = ensemble_output(rescnn_preds, transformer_preds, best_alpha)
        
        # 计算指标
        metrics = compute_comprehensive_metrics(ensemble_preds, targets)
        print_metrics(metrics, "融合模型")
        
        self.results['Ensemble'] = {
            'predictions': ensemble_preds,
            'targets': targets,
            'metrics': metrics,
            'alpha': best_alpha
        }
        
        return metrics
    
    def save_results_to_csv(self, save_dir: str = "results"):
        """保存结果到CSV文件"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 准备数据
        mse_data = []
        nmse_data = []
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            
            mse_row = {
                'Model': model_name,
                'MSE': metrics['mse'],
                'Complex_MSE': metrics['complex_mse'],
                'Amplitude_MSE': metrics['amplitude_mse'],
                'Phase_MSE': metrics['phase_mse']
            }
            mse_data.append(mse_row)
            
            nmse_row = {
                'Model': model_name,
                'NMSE': metrics['nmse'],
                'NMSE_dB': metrics['nmse_db'],
                'Amplitude_NMSE': metrics['amplitude_nmse'],
                'Correlation': metrics['correlation']
            }
            nmse_data.append(nmse_row)
        
        # 保存CSV
        pd.DataFrame(mse_data).to_csv(os.path.join(save_dir, "mse_table.csv"), index=False)
        pd.DataFrame(nmse_data).to_csv(os.path.join(save_dir, "nmse_table.csv"), index=False)
        
        print(f"结果已保存到 {save_dir} 目录")
    
    def compare_all_models(self) -> Dict[str, str]:
        """比较所有模型性能"""
        model_names = list(self.results.keys())
        metrics_list = [result['metrics'] for result in self.results.values()]
        
        comparison = compare_models(metrics_list, model_names)
        
        print(f"\n模型性能比较:")
        print("-" * 40)
        for metric, best_model in comparison.items():
            print(f"{metric}: {best_model}")
        
        return comparison

def main():
    parser = argparse.ArgumentParser(description='评估MIMO模型性能')
    
    parser.add_argument('--rescnn_checkpoint', type=str, 
                       default='checkpoints/rescnn_best.pth',
                       help='ResCNN模型检查点路径')
    parser.add_argument('--transformer_checkpoint', type=str,
                       default='checkpoints/transformer_best.pth', 
                       help='Transformer模型检查点路径')
    parser.add_argument('--batch_size', type=int, default=64, help='评估批大小')
    parser.add_argument('--optimize_ensemble', action='store_true',
                       help='是否优化融合权重')
    parser.add_argument('--save_results', action='store_true',
                       help='是否保存结果到CSV')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"评估设备: {device}")
    
    # 加载测试数据
    if os.path.exists("data/test"):
        print("加载测试数据...")
        X_test, Y_test, config = MIMODataSimulator.load_data("data/test")
        n_observed = config['n_observed']
        n_predict = config['n_predict']
    else:
        print("生成测试数据...")
        simulator = MIMODataSimulator(
            n_antennas=64,
            n_observed=16,
            n_predict=48,
            noise_std=0.1
        )
        X_test, Y_test = simulator.generate_training_data(1000)
        simulator.save_data(X_test, Y_test, "data/test")
        n_observed = 16
        n_predict = 48
    
    print(f"测试数据形状: X={X_test.shape}, Y={Y_test.shape}")
    
    # 创建测试数据加载器
    test_loader = create_data_loader(
        X_test, Y_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 创建评估器
    evaluator = ModelEvaluator(device)
    
    # 创建模型
    rescnn_model = ResCNN(
        input_dim=3,
        n_observed=n_observed,
        n_predict=n_predict,
        hidden_dim=128,
        n_layers=6
    ).to(device)
    
    transformer_model = MIMOTransformer(
        input_dim=3,
        n_observed=n_observed,
        n_predict=n_predict,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.1
    ).to(device)
    
    # 加载模型权重
    try:
        rescnn_model = evaluator.load_model_checkpoint(rescnn_model, args.rescnn_checkpoint)
        rescnn_metrics = evaluator.evaluate_single_model(rescnn_model, test_loader, "ResCNN")
    except FileNotFoundError as e:
        print(f"警告: {e}")
        rescnn_metrics = None
    
    try:
        transformer_model = evaluator.load_model_checkpoint(transformer_model, args.transformer_checkpoint)
        transformer_metrics = evaluator.evaluate_single_model(transformer_model, test_loader, "Transformer")
    except FileNotFoundError as e:
        print(f"警告: {e}")
        transformer_metrics = None
    
    # 评估融合模型
    if rescnn_metrics is not None and transformer_metrics is not None:
        ensemble_metrics = evaluator.evaluate_ensemble_model(
            rescnn_model, transformer_model, test_loader,
            optimize_weights=args.optimize_ensemble
        )
        
        # 比较所有模型
        comparison = evaluator.compare_all_models()
    
    # 保存结果
    if args.save_results:
        evaluator.save_results_to_csv()
    
    print("\n模型评估完成!")

if __name__ == "__main__":
    main()