"""
多任务学习工具函数
=================

包含:
- Frangi Filter (血管增强)
- Hessian 矩阵伪标签生成
- 图像处理工具
- 评估指标
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple
import cv2
import logging

logger = logging.getLogger(__name__)


class FrangiFilter:
    """
    Frangi 血管增强滤波器
    
    用于增强图像中的线性结构 (如血管、神经纤维)
    
    Reference:
        Frangi et al. (1998) "Multiscale Vessel Enhancement Filtering"
    """
    
    def __init__(self, sigma_range: Tuple[float, float] = (1.0, 10.0),
                 num_scales: int = 50, alpha: float = 0.5,
                 beta: float = 0.5, gamma: float = 15.0):
        """
        Parameters
        ----------
        sigma_range : Tuple[float, float]
            高斯滤波器的sigma范围
        num_scales : int
            尺度数量
        alpha, beta, gamma : float
            Frangi滤波器的参数
        """
        self.sigma_range = sigma_range
        self.num_scales = num_scales
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.sigmas = np.linspace(sigma_range[0], sigma_range[1], num_scales)
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        应用Frangi滤波器
        
        Parameters
        ----------
        image : np.ndarray
            输入图像 [H, W] 或 [H, W, C]
        
        Returns
        -------
        np.ndarray
            增强后的图像
        """
        if image.ndim == 3:
            # 转换为灰度
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        image = image / (image.max() + 1e-10)  # 归一化到[0, 1]
        
        # 存储所有尺度的响应
        frangi_responses = []
        
        for sigma in self.sigmas:
            # 计算Hessian矩阵
            Hxx = gaussian_filter(image, sigma=sigma, order=[2, 0])
            Hyy = gaussian_filter(image, sigma=sigma, order=[0, 2])
            Hxy = gaussian_filter(image, sigma=sigma, order=[1, 1])
            
            # Hessian矩阵的特征值
            trace = Hxx + Hyy

            # 判别式：对实对称 2x2 矩阵，等价于 (Hxx-Hyy)^2 + 4*Hxy^2，恒 >= 0。
            # 直接用此形式可避免 `trace**2 - 4*det` 因浮点误差出现负值导致 sqrt NaN。
            disc = (Hxx - Hyy) ** 2 + 4.0 * Hxy ** 2
            disc = np.clip(disc, 0.0, None)
            sqrt_disc = np.sqrt(disc)

            # 特征值
            lambda1 = (trace + sqrt_disc) / 2 + 1e-10
            lambda2 = (trace - sqrt_disc) / 2 + 1e-10
            
            # 确保 lambda1 >= lambda2
            lambda1, lambda2 = np.maximum(lambda1, lambda2), np.minimum(lambda1, lambda2)
            
            # Frangi滤波响应
            Ra = (lambda2 / (lambda1 + 1e-10)) ** 2
            Rb = lambda1 / (np.sqrt(np.abs(lambda1 * lambda2)) + 1e-10)
            S = np.sqrt(lambda1 ** 2 + lambda2 ** 2)
            
            response = np.exp(-Ra ** 2 / (2 * self.alpha ** 2)) * \
                      (1 - np.exp(-Rb ** 2 / (2 * self.beta ** 2))) * \
                      (1 - np.exp(-S ** 2 / (2 * self.gamma ** 2)))

            # 兜底：将 NaN / ±inf 清理为 0，避免下游阈值化产生 NaN 像素
            response = np.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)

            frangi_responses.append(response)

        # 取所有尺度的最大值
        frangi = np.max(frangi_responses, axis=0)

        return frangi.astype(np.float32)
