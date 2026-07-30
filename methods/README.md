# methods 包改进说明

## 改进概述

对 `methods` 包进行了全面的重构和优化，提高了代码的可维护性、可读性和可扩展性。

## 改进内容

### 1. 模块化重构

#### 新增文件

- **`methods/utils.py`**: 提取通用工具函数
  - `compute_id_statistics`: 计算 ID 数据统计量
  - `compute_standard_score`: 标准化分数（Z-score）
  - `get_vaod_score`: 计算 VaOD 分数
  - `fuse_norm`: 融合主分支和辅助分支分数
  - 辅助函数：`prepare_eval_model_with_id_loader`, `get_test_model_dir`, `extract_avgpool_features`, `collect_avgpool_features_with_labels`

- **`methods/norm.py`**: 特征范数计算方法
  - `l2norm`: L2 范数计算
  - `topk_l2norm`: Top-K L2 范数
  - `adaptive_topk_l2norm`: 自适应 Top-K L2 范数
  - `channel_consistency_topk`: 通道一致性 Top-K 范数
  - `adaptive_topk_channel_weight`: 自适应 Top-K 通道权重（兼容旧版 API）
  - `get_norm`: 获取模型各层特征范数
  - `get_all_activation_strength`: 获取所有样本的激活强度

#### 重构文件

- **`methods/score.py`**: OOD 评分方法
  - `get_msp_score`: Maximum Softmax Probability 评分
  - `get_energy_score`: Energy-based 评分
  - `get_topk_energy_score`: Top-K Energy 评分
  - `get_energy_entropy_score`: Energy-Entropy 融合评分
  - `get_odin_score`: ODIN 评分
  - `get_mahalanobis_score`: Mahalanobis 距离评分
  - `get_block_norm_score`: 特征块范数评分
  - `compute_mahalanobis_stats`: 计算 Mahalanobis 统计量
  - `get_score`: 统一评分接口
  - `get_all_scores`: 批量评分接口

### 2. 代码质量改进

- **类型提示**: 为所有函数添加了完整的类型提示
- **文档字符串**: 为所有函数添加了详细的文档字符串
- **代码组织**: 按功能模块化组织代码，提高可维护性
- **命名统一**: 统一函数命名规范，保持一致性

### 3. 向后兼容性

- **别名函数**: 保留 `adaptive_topk_channel_weight` 作为 `adaptive_topk_l2norm` 的别名
- **导出完整性**: 更新 `__init__.py` 确保所有函数正确导出
- **接口一致性**: 保持与现有 notebook 代码的兼容性

## 使用示例

### 基本导入

```python
from methods import (
    l2norm,
    adaptive_topk_channel_weight,
    compute_id_statistics,
    compute_standard_score,
    fuse_norm,
    get_norm,
    get_all_activation_strength,
    get_score,
    get_all_scores,
    forward_base,
    get_forward,
)
```

### 特征范数计算

```python
import torch

# 准备输入
x = torch.randn(2, 3, 32, 32)

# 计算 L2 范数
l2_result = l2norm(x)

# 计算自适应 Top-K 范数
adaptive_result = adaptive_topk_channel_weight(
    x, k_min=0.05, k_max=0.2, alpha=0.5, eps=1e-8
)
```

### 统计计算

```python
import numpy as np

# 计算 ID 统计量
scores_in = np.random.rand(3, 100)  # 3层，100个样本
means, stds, ps, pe = compute_id_statistics(scores_in)

# 标准化测试分数
scores_test = np.random.rand(3, 50)
z_scores = compute_standard_score(scores_test, means, stds)
```

### 分数融合

```python
# 融合主分支和辅助分支分数
scores_main = np.random.rand(10)
scores_other = np.random.rand(10)
fused, delta = fuse_norm(scores_main, scores_other, alpha=0.5)
```

### OOD 评分

```python
# 获取 OOD 分数
scores = get_score(
    inputs=x,
    model=model,
    forward_func=forward_base,
    method="msp",
    config=config,
    accelerator=accelerator
)

# 批量获取所有样本分数
all_scores = get_all_scores(
    model=model,
    data_loader=test_loader,
    forward_func=forward_base,
    method="energy",
    config=config
)
```

## 文件结构

```
methods/
├── __init__.py          # 包初始化和导出
├── score.py             # OOD 评分方法
├── norm.py              # 特征范数计算
├── utils.py             # 工具函数
├── forward.py           # 前向传播方法
└── get_threshold.py     # 阈值计算方法
```

## 兼容性测试

所有改进后的代码已通过兼容性测试，确保与现有 notebook（`exp0/FuseNorm_CIFAR_ResNet.ipynb` 和 `experiment/Base_CIFAR_ResNet.ipynb`）完全兼容。

测试结果：
- ✅ 所有函数导入正常
- ✅ 函数调用正确
- ✅ 输出格式符合预期
- ✅ 向后兼容性保持

## 优势

1. **更好的代码组织**: 按功能模块化，便于维护和扩展
2. **完整的类型提示**: 提高代码可读性和 IDE 支持
3. **详细的文档**: 每个函数都有清晰的文档说明
4. **向后兼容**: 确保现有代码无需修改即可使用
5. **易于扩展**: 新增功能只需在相应模块中添加即可

## 注意事项

- 使用前请确保已激活 `torchpy` conda 环境
- 所有依赖包已在 `torchpy` 环境中安装
- 如需添加新功能，建议参考现有代码结构