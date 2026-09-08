"""
多任务学习工具函数
=================

包含:
- Frangi Filter (血管增强)
- Hessian 矩阵伪标签生成
- 图像处理工具
- 评估指标
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel
from typing import Tuple, Optional, Union
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
            det = Hxx * Hyy - Hxy ** 2

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


class HessianPseudoLabeler:
    """
    基于Hessian矩阵的伪标签生成器
    
    计算Hessian矩阵的特征值来检测神经骨架
    """
    
    def __init__(self, sigma: float = 2.0, threshold: float = 0.5):
        """
        Parameters
        ----------
        sigma : float
            高斯滤波的sigma值
        threshold : float
            伪标签阈值
        """
        self.sigma = sigma
        self.threshold = threshold
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        生成伪标签
        
        Parameters
        ----------
        image : np.ndarray
            输入图像 [H, W] 或 [H, W, C]
        
        Returns
        -------
        np.ndarray
            伪标签 [H, W]，值为0或1
        """
        if image.ndim == 3:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        image = image / (image.max() + 1e-10)
        
        # 计算Hessian矩阵
        Hxx = gaussian_filter(image, sigma=self.sigma, order=[2, 0])
        Hyy = gaussian_filter(image, sigma=self.sigma, order=[0, 2])
        Hxy = gaussian_filter(image, sigma=self.sigma, order=[1, 1])
        
        # 特征值
        trace = Hxx + Hyy
        det = Hxx * Hyy - Hxy ** 2
        det = np.clip(det, 0, None)
        
        lambda1 = (trace + np.sqrt(trace ** 2 - 4 * det)) / 2 + 1e-10
        lambda2 = (trace - np.sqrt(trace ** 2 - 4 * det)) / 2 + 1e-10
        
        # 血管响应：较大的负特征值表示暗线性结构
        lambda1, lambda2 = np.maximum(lambda1, lambda2), np.minimum(lambda1, lambda2)
        
        # 伪标签：lambda2 < 0 表示检测到血管/神经
        pseudo_label = (np.abs(lambda2) > self.threshold).astype(np.uint8)
        
        return pseudo_label


def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    计算 Dice 系数
    
    Parameters
    ----------
    pred : torch.Tensor
        预测 [B, C, H, W]
    target : torch.Tensor
        目标 [B, C, H, W]
    threshold : float
        分割阈值
    
    Returns
    -------
    float
        Dice系数
    """
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    dice = 2.0 * intersection / (union + 1e-10)
    return dice.item()


def compute_iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    计算 IoU (Intersection over Union)
    
    Parameters
    ----------
    pred : torch.Tensor
        预测 [B, C, H, W]
    target : torch.Tensor
        目标 [B, C, H, W]
    threshold : float
        分割阈值
    
    Returns
    -------
    float
        IoU分数
    """
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = (pred_binary + target - pred_binary * target).sum()
    iou = intersection / (union + 1e-10)
    return iou.item()


def compute_classification_metrics(pred: torch.Tensor, target: torch.Tensor,
                                   num_classes: int) -> dict:
    """
    计算分类指标
    
    Parameters
    ----------
    pred : torch.Tensor
        预测 [B, num_classes]
    target : torch.Tensor
        目标 [B]
    num_classes : int
        类别数
    
    Returns
    -------
    dict
        包含准确率、精度、召回率的字典
    """
    pred_labels = torch.argmax(pred, dim=1)
    
    # 准确率
    accuracy = (pred_labels == target).float().mean().item()
    
    # 逐类的精度和召回率
    precisions = []
    recalls = []
    
    for cls in range(num_classes):
        tp = ((pred_labels == cls) & (target == cls)).sum().float()
        fp = ((pred_labels == cls) & (target != cls)).sum().float()
        fn = ((pred_labels != cls) & (target == cls)).sum().float()
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        
        precisions.append(precision.item())
        recalls.append(recall.item())
    
    return {
        'accuracy': accuracy,
        'macro_precision': np.mean(precisions),
        'macro_recall': np.mean(recalls),
        'macro_f1': 2 * np.mean(precisions) * np.mean(recalls) / (np.mean(precisions) + np.mean(recalls) + 1e-10),
    }


def batch_pseudo_labels(images: np.ndarray, method: str = 'frangi') -> torch.Tensor:
    """
    批量生成伪标签
    
    Parameters
    ----------
    images : np.ndarray
        输入图像批次 [B, H, W, 3]
    method : str
        方法: 'frangi' 或 'hessian'
    
    Returns
    -------
    torch.Tensor
        伪标签 [B, 1, H, W]
    """
    if method == 'frangi':
        labeler = FrangiFilter()
    elif method == 'hessian':
        labeler = HessianPseudoLabeler()
    else:
        raise ValueError(f"未知的方法: {method}")
    
    pseudo_labels = []
    for img in images:
        label = labeler(img)
        pseudo_labels.append(label)
    
    # 转换为张量
    pseudo_labels = torch.from_numpy(np.stack(pseudo_labels)).unsqueeze(1).float()
    return pseudo_labels


class MultiTaskLoss(nn.Module):
    """
    多任务学习损失函数
    
    加权组合所有任务的损失
    """
    
    def __init__(self, weights: dict = None):
        """
        Parameters
        ----------
        weights : dict
            各任务损失的权重
        """
        super().__init__()
        
        if weights is None:
            weights = {
                'seg': 0.3,
                'mae': 0.2,
                'pseudo': 0.1,
                'pd': 0.2,
                'stage': 0.1,
                'subtype': 0.1,
            }
        
        # 归一化权重
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
    
    def forward(self, losses: dict) -> torch.Tensor:
        """
        计算加权总损失
        
        Parameters
        ----------
        losses : dict
            各任务的损失
        
        Returns
        -------
        torch.Tensor
            加权总损失
        """
        total_loss = 0.0
        
        for key, weight in self.weights.items():
            if f'seg_{key}' in losses or f'cls_{key}' in losses:
                if f'seg_{key}' in losses:
                    total_loss += weight * losses[f'seg_{key}']
                if f'cls_{key}' in losses:
                    total_loss += weight * losses[f'cls_{key}']
        
        return total_loss


if __name__ == '__main__':
    # 测试 Frangi Filter
    print("测试 Frangi Filter...")
    frangi = FrangiFilter()
    
    # 创建测试图像
    test_image = np.random.rand(256, 256).astype(np.float32) * 255
    result = frangi(test_image)
    print(f"输入形状: {test_image.shape}")
    print(f"输出形状: {result.shape}")
    print(f"输出范围: [{result.min():.3f}, {result.max():.3f}]")
    
    # 测试 Hessian 伪标签生成
    print("\n测试 Hessian 伪标签...")
    hessian = HessianPseudoLabeler()
    pseudo_label = hessian(test_image)
    print(f"伪标签形状: {pseudo_label.shape}")
    print(f"伪标签值: {np.unique(pseudo_label)}")
