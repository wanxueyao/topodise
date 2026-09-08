"""
skeletonize.py - 神经纤维图像骨架提取

将 UNet 分割后的二值神经纤维图像细化为单像素宽的骨架，
作为图构建的基础输入。

依赖: scikit-image (pip install scikit-image)
"""

import numpy as np
from PIL import Image


def load_binary_image(image_path: str, threshold: int = 128) -> np.ndarray:
    """
    加载图像并转为二值 mask。

    Args:
        image_path: 图像文件路径
        threshold: 二值化阈值 (大于该值视为神经纤维)

    Returns:
        二值 numpy 数组 [H, W], dtype bool
    """
    img = Image.open(image_path).convert('L')
    arr = np.array(img)
    return arr > threshold


def skeletonize_image(binary: np.ndarray) -> np.ndarray:
    """
    对二值图像进行形态学骨架化，将神经纤维细化为单像素宽的骨架。

    Args:
        binary: 二值 [H, W] 数组 (True = 神经纤维)

    Returns:
        骨架 [H, W] 数组, dtype bool
    """
    from skimage.morphology import skeletonize
    return skeletonize(binary)


def image_to_skeleton(image_path: str, threshold: int = 128):
    """
    完整流程: 图像文件 → (二值 mask, 骨架)。

    Args:
        image_path: 神经纤维图像路径
        threshold: 二值化阈值

    Returns:
        (binary, skeleton): 均为 [H, W] bool 数组
    """
    binary = load_binary_image(image_path, threshold)
    skeleton = skeletonize_image(binary)
    return binary, skeleton
