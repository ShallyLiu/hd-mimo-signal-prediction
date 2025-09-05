"""
训练工具函数
包含损失函数、优化器配置、训练循环等工具
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os
from typing import Dict, List, Tuple, Optional, Callable

class ComplexMSELoss(nn.Module):
    """复数信号MSE损失函数"""
    
    def __init__(self):
        super(ComplexMSELoss, self).__init__()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算复数信号的MSE损失
        
        Args:
            pred: 预测值 [batch_size, n_predict, 2] (Re/Im)
            target: 真实值 [batch_size, n_predict, 2] (Re/Im)
        """
        mse_loss = nn.functional.mse_loss(pred, target, reduction='mean')
        return mse_loss

class NMSELoss(nn.Module):
    """归一化均方误差损失函数"""
    
    def __init__(self):
        super(NMSELoss, self).__init__()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算NMSE损失
        
        Args:
            pred: 预测值 [batch_size, n_predict, 2]
            target: 真实值 [batch_size, n_predict, 2]
        """
        # 计算复数模长
        pred_complex = pred[:, :, 0] + 1j * pred[:, :, 1]
        target_complex = target[:, :, 0] + 1j * target[:, :, 1]
        
        # 计算NMSE
        numerator = torch.mean(torch.abs(pred_complex - target_complex) ** 2)
        denominator = torch.mean(torch.abs(target_complex) ** 2)
        
        nmse = numerator / (denominator + 1e-8)
        return nmse

def create_data_loader(X: np.ndarray, 
                      Y: np.ndarray, 
                      batch_size: int = 64,
                      shuffle: bool = True,
                      num_workers: int = 0) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        X: 输入数据
        Y: 标签数据
        batch_size: 批大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
    """
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(Y).float()
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

def configure_optimizer(model: nn.Module,
                       lr: float = 1e-3,
                       weight_decay: float = 1e-4,
                       optimizer_type: str = 'adamw') -> optim.Optimizer:
    """
    配置优化器
    
    Args:
        model: 模型
        lr: 学习率
        weight_decay: 权重衰减
        optimizer_type: 优化器类型
    """
    if optimizer_type.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")

def configure_scheduler(optimizer: optim.Optimizer,
                       scheduler_type: str = 'cosine',
                       T_max: int = 100,
                       eta_min: float = 1e-6) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    配置学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型
        T_max: 余弦退火的最大epoch数
        eta_min: 最小学习率
    """
    if scheduler_type.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_type.lower() == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type.lower() == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    elif scheduler_type.lower() == 'none':
        return None
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            val_loss: 验证损失
            
        Returns:
            是否应该早停
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 save_dir: str = "checkpoints"):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_nmse = []
        self.val_nmse = []
        
    def train_epoch(self,
                   optimizer: optim.Optimizer,
                   criterion: nn.Module,
                   nmse_criterion: nn.Module) -> Tuple[float, float]:
        """训练一个epoch"""
        
        self.model.train()
        total_loss = 0.0
        total_nmse = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            # 前向传播
            output = self.model(data)
            
            # 计算损失
            loss = criterion(output, target)
            nmse = nmse_criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_nmse += nmse.item()
            num_batches += 1
            
        return total_loss / num_batches, total_nmse / num_batches
    
    def validate_epoch(self,
                      criterion: nn.Module,
                      nmse_criterion: nn.Module) -> Tuple[float, float]:
        """验证一个epoch"""
        
        self.model.eval()
        total_loss = 0.0
        total_nmse = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                
                loss = criterion(output, target)
                nmse = nmse_criterion(output, target)
                
                total_loss += loss.item()
                total_nmse += nmse.item()
                num_batches += 1
                
        return total_loss / num_batches, total_nmse / num_batches
    
    def train(self,
              num_epochs: int,
              lr: float = 1e-3,
              weight_decay: float = 1e-4,
              optimizer_type: str = 'adamw',
              scheduler_type: str = 'cosine',
              early_stopping_patience: int = 20,
              save_best: bool = True,
              model_name: str = "model") -> Dict[str, List[float]]:
        """
        完整训练流程
        
        Args:
            num_epochs: 训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            optimizer_type: 优化器类型
            scheduler_type: 调度器类型
            early_stopping_patience: 早停耐心值
            save_best: 是否保存最佳模型
            model_name: 模型名称
        """
        print(f"开始训练 {model_name} 模型...")
        print(f"训练设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 配置组件
        criterion = ComplexMSELoss().to(self.device)
        nmse_criterion = NMSELoss().to(self.device)
        optimizer = configure_optimizer(self.model, lr, weight_decay, optimizer_type)
        scheduler = configure_scheduler(optimizer, scheduler_type, num_epochs)
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_loss, train_nmse = self.train_epoch(optimizer, criterion, nmse_criterion)
            
            # 验证
            val_loss, val_nmse = self.validate_epoch(criterion, nmse_criterion)
            
            # 更新学习率
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_nmse.append(train_nmse)
            self.val_nmse.append(val_nmse)
            
            # 保存最佳模型
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_nmse': train_nmse,
                    'val_nmse': val_nmse
                }, os.path.join(self.save_dir, f"{model_name}_best.pth"))
            
            # 打印进度
            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] - "
                      f"训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}, "
                      f"训练NMSE: {train_nmse:.6f}, 验证NMSE: {val_nmse:.6f}, "
                      f"学习率: {current_lr:.2e}, 耗时: {epoch_time:.2f}s")
            
            # 早停检查
            if early_stopping(val_loss):
                print(f"第 {epoch+1} 轮触发早停")
                break
        
        total_time = time.time() - start_time
        print(f"训练完成! 总耗时: {total_time:.2f}s")
        print(f"最佳验证损失: {best_val_loss:.6f}")
        
        # 保存最终模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_nmse': self.train_nmse,
            'val_nmse': self.val_nmse
        }, os.path.join(self.save_dir, f"{model_name}_final.pth"))
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_nmse': self.train_nmse,
            'val_nmse': self.val_nmse
        }