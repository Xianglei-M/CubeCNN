import numpy as np
import torch
import torch


class ClassificationMetric:
    def __init__(self, num_classes, ignore_index=None):
        """
        Args:
            num_classes: 类别数
            ignore_index: 需要忽略的标签（如 255），None 表示不忽略
            device: 混淆矩阵存放的设备，'cpu' 或 'cuda'
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confusion_matrix = torch.zeros(
            (num_classes, num_classes),
            dtype=torch.long,
            device=self.device
        )

    def _fast_hist(self, pred, gt):
        """
        pred: (N,) long tensor
        gt:   (N,) long tensor
        """
        # 有效像素掩码：标签在 [0, num_classes) 且不等于 ignore_index
        mask = (gt >= 0) & (gt < self.num_classes)
        if self.ignore_index is not None:
            mask = mask & (gt != self.ignore_index)

        pred_valid = pred[mask].long()
        gt_valid = gt[mask].long()

        # 计算混淆矩阵：行是 GT，列是 Pred
        indices = self.num_classes * gt_valid + pred_valid
        hist = torch.bincount(
            indices,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        return hist

    def update(self, pred, gt):
        """
        pred: (N,) / (B, N) 或 (B, H, W) 等任意形状
        gt:   与 pred 形状相同
        """
        if not isinstance(pred, torch.Tensor):
            pred = torch.as_tensor(pred)
        if not isinstance(gt, torch.Tensor):
            gt = torch.as_tensor(gt)

        # 搬到混淆矩阵所在设备
        '''
        pred = pred.to(self.device).reshape(-1)
        gt = gt.to(self.device).reshape(-1)
        '''
        pred = pred.reshape(-1)
        gt = gt.reshape(-1)

        # 累加到混淆矩阵（保证 dtype 一致）
        self.confusion_matrix += self._fast_hist(pred, gt)

    def reset(self):
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.long,
            device=self.device
        )

    def get_metrics(self):
        cm = self.confusion_matrix.float()

        # TP
        TP = torch.diag(cm)

        # FP
        FP = cm.sum(dim=0) - TP

        # FN
        FN = cm.sum(dim=1) - TP

        # TN（一般不用）
        TN = cm.sum() - (TP + FP + FN)

        eps = 1e-6

        # IoU
        IoU = TP / (TP + FP + FN + eps)

        # Precision
        Precision = TP / (TP + FP + eps)

        # Recall
        Recall = TP / (TP + FN + eps)

        # F1
        F1 = 2 * Precision * Recall / (Precision + Recall + eps)

        # Overall Accuracy
        OA = TP.sum() / (cm.sum() + eps)

        # Mean Accuracy
        Acc = TP / (cm.sum(dim=1) + eps)
        mAcc = Acc.mean()

        # Mean IoU
        mIoU = IoU.mean()

        return {
            "OA": OA.item(),
            "mAcc": mAcc.item(),
            "mIoU": mIoU.item(),
            "IoU": IoU.cpu().numpy(),
            "Precision": Precision.cpu().numpy(),
            "Recall": Recall.cpu().numpy(),
            "F1": F1.cpu().numpy(),
            "confusion_matrix": cm.cpu().numpy()
        }
