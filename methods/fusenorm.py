#!/usr/bin/env python3
"""
FuseNorm: A method for fusing scores from strong and weak experts for OOD detection.

This module provides a clean, reusable implementation of the FuseNorm method.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Union


def _df(x: np.ndarray) -> np.ndarray:
    """
    Nonlinear deviation scaling function.
    
    Args:
        x: 输入数组
        
    Returns:
        缩放后的结果
    """
    return np.minimum(1, np.power(20, x) - 1)


def fuse_norm(
    scores_main: np.ndarray,
    scores_other: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse scores from strong and weak experts (FuseNorm).

    Args:
        scores_main: np.array, strong expert scores [N] or [L_strong, N]
        scores_other: np.array, weak experts' normalized scores [L_weak, N]

    Returns:
        fused_score: np.array, adjusted strong expert scores [1, N]
        delta_score: np.array, OOD confidence adjustment [1, N]
    """
    scores_main = np.atleast_2d(np.copy(scores_main))
    scores_other = np.copy(scores_other)

    # Mean strong expert score across multiple strong cues (if any)
    scores_main_mean = np.mean(scores_main, axis=0, keepdims=True)

    # Compute deviation for weak experts
    s_high = scores_other > 1
    s_low = scores_other < 0
    scores_other = np.abs(scores_other * s_low + (scores_other - 1) * s_high)

    # Take max deviation among weak experts
    scores_other = np.max(scores_other, axis=0, keepdims=True)

    # Scale deviation (confidence calibration)
    delta_score = _df(scores_other)

    # Fuse: subtract deviation from strong expert
    fused_score = scores_main_mean - delta_score

    return fused_score, delta_score




# 定义公开导出的接口
__all__ = [ "fuse_norm"]
