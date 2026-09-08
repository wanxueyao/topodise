import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_training_curves(history, task_name, fold_idx, output_dir):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Linux 通用字体：DejaVu Sans 几乎所有 Linux 发行版都预装
    # Liberation Sans 是 Red Hat 系列的替代字体
    # Noto Sans CJK 支持中日韩字符（如需中文标签）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Noto Sans CJK SC', 'Noto Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Loss曲线
    axes[0, 0].plot(history['train_loss'], label='Train', lw=2)
    axes[0, 0].plot(history['val_loss'], label='Val', lw=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'{task_name} - Loss (Fold {fold_idx})')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Accuracy曲线
    axes[0, 1].plot(history['train_acc'], label='Train', lw=2)
    axes[0, 1].plot(history['val_acc'], label='Val', lw=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title(f'{task_name} - Accuracy (Fold {fold_idx})')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # F1 曲线
    axes[1, 0].plot(history['val_f1'], label='Val F1', lw=2, marker='o', ms=3)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title(f'{task_name} - F1 (Fold {fold_idx})')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 最佳指标文本
    axes[1, 1].axis('off')
    text = f"""
    Best Val Accuracy: {history['best_val_acc']:.4f}
    Final Val Accuracy: {history['val_acc'][-1]:.4f}
    Final Val F1: {history['val_f1'][-1]:.4f}
    """
    axes[1, 1].text(0.1, 0.5, text, fontsize=11, va='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    safe_name = task_name.replace('/', '_').replace('\\', '_')
    plt.savefig(f'{output_dir}/{safe_name}_fold{fold_idx}_curves.png', dpi=200)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, task_name, fold_idx, output_dir, is_test=False):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    title = f'{task_name} - Confusion Matrix (Test)' if is_test else f'{task_name} - CM (Fold {fold_idx})'
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    suffix = 'Test_cm' if is_test else f'fold{fold_idx}_cm'
    plt.savefig(f'{output_dir}/{task_name}_{suffix}.png', dpi=200)
    plt.close()


def plot_roc_auc(y_true, y_probs, class_names, task_name, output_dir):
    """绘制ROC曲线并返回各类别AUC"""
    y_probs = np.asarray(y_probs)
    y_true = np.asarray(y_true)
    if y_probs.ndim != 2 or y_probs.shape[0] == 0:
        return {}

    n_classes = min(len(class_names), y_probs.shape[1])
    if n_classes < 2:
        return {}
    class_names = list(class_names)[:n_classes]
    
    if n_classes == 2:
        if np.unique(y_true).size < 2:
            return {}
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(7, 7))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0, 1])
        plt.ylim([0, 1.02])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'{task_name} - ROC')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{task_name}_Test_roc.png', dpi=200)
        plt.close()
        
        return roc_auc
    else:
        from sklearn.preprocessing import label_binarize
        
        y_bin = label_binarize(y_true, classes=range(n_classes))
        fpr, tpr, roc_auc = {}, {}, {}
        
        for i in range(n_classes):
            # 某类在当前评估集中可能全是 0 或全是 1，ROC 不可定义；跳过该类。
            if np.unique(y_bin[:, i]).size < 2:
                continue
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        if not roc_auc:
            return {}
        
        plt.figure(figsize=(7, 7))
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            if i not in roc_auc:
                continue
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{class_names[i]} (AUC={roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0, 1])
        plt.ylim([0, 1.02])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'{task_name} - ROC (Per Class)')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{task_name}_Test_roc.png', dpi=200)
        plt.close()
        
        return roc_auc
