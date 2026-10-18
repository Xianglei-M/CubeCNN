import numpy as np
import torch

class ShapeNetIoU:
    # def __init__(self, num_classes, ignore_index=None, seg_classes=None):
    def __init__(self, num_classes=None):
        self.num_classes = num_classes
        # self.ignore_index = ignore_index
        # self.confusion_matrix = np.zeros((num_classes, num_classes))

        # self.seg_classes = seg_classes
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
        '''
        self.seg_classes_2 = {'Airplane': [0, 1, 2, 3], 'Bag': [4, 5], 'Cap': [6, 7], 'Car': [8, 9, 10, 11],
                              'Chair': [12, 13, 14, 15], 'Earphone': [16, 17, 18], 'Guitar': [19, 20, 21],
                              'Knife': [22, 23], 'Lamp': [24, 25, 26, 27], 'Laptop': [28, 29],
                              'Motorbike': [30, 31, 32, 33, 34, 35], 'Mug': [36, 37], 'Pistol': [38, 39, 40],
                              'Rocket': [41, 42, 43], 'Skateboard': [44, 45, 46], 'Table': [47, 48, 49]}
        '''

        self.seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in self.seg_classes.keys():
            for label in self.seg_classes[cat]:
                self.seg_label_to_cat[label] = cat
        self.shape_ious_all = []
        self.shape_ious_classed = {cat: [] for cat in self.seg_classes.keys()}
        self.class_ious = {cat: 0 for cat in self.seg_classes.keys()}

    '''
        def _fast_hist(self, pred, gt):
        """
        pred: (N,)
        gt:   (N,)
        """
        mask = (gt >= 0) & (gt < self.num_classes)
        if self.ignore_index is not None:
            mask = mask & (gt != self.ignore_index)

        hist = np.bincount(
            self.num_classes * gt[mask].astype(int) + pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        return hist
    '''

    def update(self, pred, gt):
        """
        pred: (B, N)
        gt:   (B, N)
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()

        cur_batch_size, _ = gt.shape
        for i in range(cur_batch_size):
            segp = pred[i, :]
            segl = gt[i, :]
            cat = self.seg_label_to_cat[segl[0]]
            # part_ious = [0.0 for _ in range(len(seg_classes[cat]))]  # part-class-IoU
            part_ious = []
            for l in self.seg_classes[cat]:
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    # part_ious[l - seg_classes[cat][0]] = 1.0
                    part_iou = 1.0
                    part_ious.append(part_iou)
                else:
                    # part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                    #     np.sum((segl == l) | (segp == l)))
                    part_iou = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
                    part_ious.append(part_iou)
            self.shape_ious_classed[cat].append(np.mean(part_ious))

    def reset(self):
        # self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.shape_ious_all = []
        self.shape_ious_classed = {cat: [] for cat in self.seg_classes.keys()}

    '''
        def get_metrics(self):
        cm = self.confusion_matrix

        # TP
        TP = np.diag(cm)

        # FP
        FP = cm.sum(axis=0) - TP

        # FN
        FN = cm.sum(axis=1) - TP

        # TN（一般不用）
        TN = cm.sum() - (TP + FP + FN)

        # IoU
        IoU = TP / (TP + FP + FN + 1e-6)

        # Precision
        Precision = TP / (TP + FP + 1e-6)

        # Recall
        Recall = TP / (TP + FN + 1e-6)

        # F1
        F1 = 2 * Precision * Recall / (Precision + Recall + 1e-6)

        # Overall Accuracy
        OA = TP.sum() / (cm.sum() + 1e-6)

        # Mean Accuracy
        Acc = TP / (cm.sum(axis=1) + 1e-6)
        mAcc = np.mean(Acc)

        # Mean IoU
        mIoU = np.mean(IoU)

        return {
            "OA": OA,
            "mAcc": mAcc,
            "mIoU": mIoU,
            "IoU": IoU,
            "Precision": Precision,
            "Recall": Recall,
            "F1": F1,
            "confusion_matrix": cm
        }
    '''

    def get_statistics(self):
        # shape_iou_classed = {cat: 0.0 for cat in self.seg_classes.keys()}
        for cat in self.shape_ious_classed.keys():
            for iou in self.shape_ious_classed[cat]:
                self.shape_ious_all.append(iou)
            # shape_ious[cat] = np.mean(shape_ious[cat])  # 此时，shape_ious: mIoU = mean(all的class-mIoU)
            self.class_ious[cat] = np.mean(self.shape_ious_classed[cat])

        class_avg_iou = np.mean(list(self.class_ious.values()))
        all_shape_avg_iou = np.mean(self.shape_ious_all)
        '''
        mean_shape_ious = np.mean(list(shape_ious.values()))  # mIoU = mean(all的part-class-mIoU)
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
        for cat in sorted(shape_ious.keys()):
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
        '''

        return {
            "class_avg_iou": class_avg_iou,
            # "all_shape_avg_iou": all_shape_avg_iou,
            "instance_avg_iou": all_shape_avg_iou,
        }