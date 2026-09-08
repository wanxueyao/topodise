"""
多任务学习推理和评估脚本
=======================

用于模型推理、评估和可视化结果
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict
import logging

from .unet_multitask import UNetMultiTask
from .multitask_utils import (
    compute_dice_score,
    compute_iou_score,
    compute_classification_metrics,
)

logger = logging.getLogger(__name__)


class MTLInference:
    """多任务学习推理器"""

    def __init__(self, model: UNetMultiTask, device: str = "cuda"):
        """
        Parameters
        ----------
        model : UNetMultiTask
            训练好的模型
        device : str
            设备
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def infer_segmentation(self, images: torch.Tensor) -> torch.Tensor:
        """
        推理分割

        Parameters
        ----------
        images : torch.Tensor
            输入图像 [B, 3, H, W]

        Returns
        -------
        torch.Tensor
            分割掩模 [B, 1, H, W]
        """
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images, training=False, use_mae=False)
            segmentation = outputs["segmentation"]
        return segmentation

    def infer_classification(self, images: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        推理分类

        Parameters
        ----------
        images : torch.Tensor
            输入图像 [B, 3, H, W]

        Returns
        -------
        Dict[str, np.ndarray]
            分类结果
        """
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images, training=False, use_mae=False)

            pd_probs = F.softmax(outputs["pd_logits"], dim=1).cpu().numpy()
            stage_probs = F.softmax(outputs["stage_logits"], dim=1).cpu().numpy()
            subtype_probs = F.softmax(outputs["subtype_logits"], dim=1).cpu().numpy()

        return {
            "pd_probs": pd_probs,
            "stage_probs": stage_probs,
            "subtype_probs": subtype_probs,
            "pd_pred": np.argmax(pd_probs, axis=1),
            "stage_pred": np.argmax(stage_probs, axis=1),
            "subtype_pred": np.argmax(subtype_probs, axis=1),
        }

    def infer_joint(self, images: torch.Tensor) -> Dict:
        """
        联合推理分割和分类

        Parameters
        ----------
        images : torch.Tensor
            输入图像 [B, 3, H, W]

        Returns
        -------
        Dict
            包含分割和分类结果的字典
        """
        seg = self.infer_segmentation(images)
        cls = self.infer_classification(images)

        return {
            "segmentation": seg,
            "classification": cls,
        }


class MTLEvaluator:
    """多任务学习评估器"""

    def __init__(self, model: UNetMultiTask, device: str = "cuda"):
        """
        Parameters
        ----------
        model : UNetMultiTask
            模型
        device : str
            设备
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def evaluate_segmentation(self, val_loader) -> Dict[str, float]:
        """
        评估分割性能

        Parameters
        ----------
        val_loader : DataLoader
            验证数据加载器

        Returns
        -------
        Dict[str, float]
            评估指标
        """
        dice_scores = []
        iou_scores = []

        with torch.no_grad():
            for batch in val_loader:
                if "mask" not in batch:
                    continue

                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                outputs = self.model(images, training=False, use_mae=False)
                seg = outputs["segmentation"]

                dice = compute_dice_score(seg, masks)
                iou = compute_iou_score(seg, masks)

                dice_scores.append(dice)
                iou_scores.append(iou)

        return {
            "mean_dice": np.mean(dice_scores),
            "std_dice": np.std(dice_scores),
            "mean_iou": np.mean(iou_scores),
            "std_iou": np.std(iou_scores),
        }

    def evaluate_classification(self, val_loader) -> Dict[str, float]:
        """
        评估分类性能

        Parameters
        ----------
        val_loader : DataLoader
            验证数据加载器

        Returns
        -------
        Dict[str, float]
            评估指标
        """
        pd_preds = []
        pd_labels = []

        with torch.no_grad():
            for batch in val_loader:
                if "pd_label" not in batch:
                    continue

                images = batch["image"].to(self.device)
                labels = batch["pd_label"].to(self.device)

                outputs = self.model(images, training=False, use_mae=False)
                pd_logits = outputs["pd_logits"]

                pd_preds.append(pd_logits.cpu())
                pd_labels.append(labels.cpu())

        if not pd_preds:
            return {}

        pd_preds = torch.cat(pd_preds, dim=0)
        pd_labels = torch.cat(pd_labels, dim=0)

        metrics = compute_classification_metrics(pd_preds, pd_labels, 2)

        return {
            "accuracy": metrics["accuracy"],
            "precision": metrics["macro_precision"],
            "recall": metrics["macro_recall"],
            "f1": metrics["macro_f1"],
        }

    def evaluate_joint(self, seg_loader, cls_loader) -> Dict:
        """
        联合评估

        Parameters
        ----------
        seg_loader : DataLoader
            分割验证加载器
        cls_loader : DataLoader
            分类验证加载器

        Returns
        -------
        Dict
            评估结果
        """
        seg_metrics = self.evaluate_segmentation(seg_loader)
        cls_metrics = self.evaluate_classification(cls_loader)

        return {
            "segmentation": seg_metrics,
            "classification": cls_metrics,
        }


class ModelSelector:
    """
    模型选择和超参数调优

    支持多个模型的快速比较
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.models = {}
        self.results = {}

    def add_model(self, name: str, config: Dict):
        """
        添加模型配置

        Parameters
        ----------
        name : str
            模型名称
        config : Dict
            模型配置
        """
        model = UNetMultiTask(**config)
        self.models[name] = model

    def evaluate_all(self, test_loader) -> Dict:
        """
        评估所有模型

        Parameters
        ----------
        test_loader : DataLoader
            测试数据加载器

        Returns
        -------
        Dict
            所有模型的评估结果
        """
        for name, model in self.models.items():
            evaluator = MTLEvaluator(model, self.device)

            # 分类评估
            cls_metrics = evaluator.evaluate_classification(test_loader)

            self.results[name] = cls_metrics

            logger.info(f"模型 {name} - 准确率: {cls_metrics.get('accuracy', 0):.4f}")

        return self.results

    def get_best_model(self) -> str:
        """
        获取性能最佳的模型

        Returns
        -------
        str
            最佳模型名称
        """
        best_name = max(self.results, key=lambda x: self.results[x].get("accuracy", 0))
        return best_name


if __name__ == "__main__":
    # 测试推理和评估
    from .config import MTLConfig

    logger.info("测试推理和评估...")

    config = MTLConfig()
    model = UNetMultiTask()

    # 测试推理
    inference = MTLInference(model, "cpu")
    images = torch.randn(2, 3, 256, 256)

    seg = inference.infer_segmentation(images)
    print(f"分割输出: {seg.shape}")

    cls = inference.infer_classification(images)
    print(f"分类输出 - PD: {cls['pd_pred']}")
