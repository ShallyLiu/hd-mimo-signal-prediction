"""
模型融合模块
提供多种模型融合策略
"""

from .fusion import (
    ensemble_output,
    weighted_ensemble,
    AdaptiveEnsemble,
    VarianceBasedEnsemble,
    EnsembleOptimizer,
    evaluate_ensemble_performance
)

__all__ = [
    'ensemble_output',
    'weighted_ensemble', 
    'AdaptiveEnsemble',
    'VarianceBasedEnsemble',
    'EnsembleOptimizer',
    'evaluate_ensemble_performance'
]