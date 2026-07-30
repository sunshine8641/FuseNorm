"""
methods 包

提供 OOD 检测相关的评分方法和特征处理工具。
"""

# 从 score.py 导入 OOD 评分方法
from .score import (
    get_score,
    get_all_scores,
    compute_mahalanobis_stats,
    get_msp_score,
    get_energy_score,
    get_topk_energy_score,
    get_energy_entropy_score,
    get_odin_score,
    get_mahalanobis_score,
    get_block_norm_score,
    get_cadref_score,
)

# 从 norm.py 导入特征范数计算方法
from .norm import (
    l2norm,
    topk_l2norm,
    adaptive_topk_channel_weight,
    get_norm,
    get_all_activation_strength,
)

# 从 utils.py 导入工具函数
from .utils import (
    compute_id_statistics,
    compute_standard_score,
    # get_vaod_score,
)

# 从 fusenorm.py 导入 FuseNorm 相关方法
from .fusenorm import (
    fuse_norm,
)

# 从 forward.py 导入前向传播方法
from .forward import (
    forward_base,
    forward_all_features,
    forward_react,
    forward_bats,
    forward_laps,
    forward_cadref,
    get_forward,
)

# 从 get_threshold.py 导入阈值计算方法
from .get_threshold import (
    compute_threshold_react,
    compute_activation_stats,
    compute_cadref_stats,
    calculate_layer_norm,
)

# 定义公开导出的接口
__all__ = [
    # OOD 评分方法
    'get_score',
    'get_all_scores',
    'compute_mahalanobis_stats',
    'get_msp_score',
    'get_energy_score',
    'get_topk_energy_score',
    'get_energy_entropy_score',
    'get_odin_score',
    'get_mahalanobis_score',
    'get_block_norm_score',
    
    # 特征范数计算
    'l2norm',
    'topk_l2norm',
    'adaptive_topk_channel_weight',
    'get_norm',
    'get_all_activation_strength',
    
    # 工具函数
    'compute_id_statistics',
    'compute_standard_score',
    # 'get_vaod_score',
    'fuse_norm',
    
    # 前向传播方法
    'forward_base',
    'forward_all_features',
    'forward_react',
    'forward_bats',
    'forward_laps',
    'forward_cadref',
    'get_forward',

    # 阈值计算
    'compute_threshold_react',
    'compute_activation_stats',
    'compute_cadref_stats',
    'calculate_layer_norm',

    # CADRef
    'get_cadref_score',
]