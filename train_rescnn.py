"""
ResCNN模型训练脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import argparse
from models.rescnn import ResCNN
from train.utils import ModelTrainer, create_data_loader
from data_simulator import MIMODataSimulator

def train_rescnn(config):
    """训练ResCNN模型"""
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # 针对RTX3090优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"使用GPU: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("使用CPU训练")
    
    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    # 加载或生成数据
    if os.path.exists("data/train"):
        print("加载已有数据...")
        X_train, Y_train, train_config = MIMODataSimulator.load_data("data/train")
        X_val, Y_val, _ = MIMODataSimulator.load_data("data/val")
        n_observed = train_config['n_observed']
        n_predict = train_config['n_predict']
    else:
        print("生成新数据...")
        simulator = MIMODataSimulator(
            n_antennas=config['n_antennas'],
            n_observed=config['n_observed'],
            n_predict=config['n_predict'],
            noise_std=config['noise_std']
        )
        
        # 生成数据
        X_train, Y_train = simulator.generate_training_data(config['train_samples'])
        X_val, Y_val = simulator.generate_training_data(config['val_samples'])
        
        # 保存数据
        simulator.save_data(X_train, Y_train, "data/train")
        simulator.save_data(X_val, Y_val, "data/val")
        
        n_observed = config['n_observed']
        n_predict = config['n_predict']
    
    print(f"训练数据形状: X={X_train.shape}, Y={Y_train.shape}")
    print(f"验证数据形状: X={X_val.shape}, Y={Y_val.shape}")
    
    # 创建数据加载器
    train_loader = create_data_loader(
        X_train, Y_train, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4
    )
    
    val_loader = create_data_loader(
        X_val, Y_val, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=4
    )
    
    # 创建模型
    model = ResCNN(
        input_dim=3,
        n_observed=n_observed,
        n_predict=n_predict,
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers']
    ).to(device)
    
    print(f"ResCNN模型参数数量: {model.get_model_size():,}")
    
    # 创建训练器
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir="checkpoints"
    )
    
    # 开始训练
    training_history = trainer.train(
        num_epochs=config['num_epochs'],
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        optimizer_type=config['optimizer'],
        scheduler_type=config['scheduler'],
        early_stopping_patience=config['early_stopping_patience'],
        save_best=True,
        model_name="rescnn"
    )
    
    # 保存训练历史
    np.save("checkpoints/rescnn_training_history.npy", training_history)
    
    return training_history

def main():
    parser = argparse.ArgumentParser(description='训练ResCNN模型')
    
    # 数据参数
    parser.add_argument('--n_antennas', type=int, default=64, help='总天线数量')
    parser.add_argument('--n_observed', type=int, default=16, help='观测天线数量')
    parser.add_argument('--n_predict', type=int, default=48, help='预测天线数量')
    parser.add_argument('--noise_std', type=float, default=0.1, help='噪声标准差')
    parser.add_argument('--train_samples', type=int, default=10000, help='训练样本数')
    parser.add_argument('--val_samples', type=int, default=2000, help='验证样本数')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--n_layers', type=int, default=6, help='残差块层数')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--optimizer', type=str, default='adamw', help='优化器')
    parser.add_argument('--scheduler', type=str, default='cosine', help='学习率调度器')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='早停耐心值')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'n_antennas': args.n_antennas,
        'n_observed': args.n_observed,
        'n_predict': args.n_predict,
        'noise_std': args.noise_std,
        'train_samples': args.train_samples,
        'val_samples': args.val_samples,
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'early_stopping_patience': args.early_stopping_patience,
        'seed': args.seed
    }
    
    print("=" * 50)
    print("ResCNN模型训练配置:")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    # 开始训练
    training_history = train_rescnn(config)
    
    print("ResCNN模型训练完成!")

if __name__ == "__main__":
    main()