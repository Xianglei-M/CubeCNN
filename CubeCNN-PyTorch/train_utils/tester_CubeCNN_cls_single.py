import torch
from torch import nn
from tqdm import tqdm
import provider

from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.data_augmentation import DataAugmentation

from models.CubeCNN_cls import CubeCNN

from statistics_utils.segmentation_metric_util import ClassificationMetric
from statistics_utils.shapenet_partseg_statistics_util import ShapeNetIoU


class ModelTester:
    def __init__(self, args, savepath):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.args = args
        self.save_path = savepath

        # self.augment = PointCloudAugment(args.normal)

        self.num_class = 10
        if args.normal:
            additional_channel = 3
        else:
            additional_channel = 0

        parameters_list = []
        params = {
            'in_channel' :  3,  # in_channel = lvconv_list[-1] * (n_layers + 1)
            'split_list': [4, 4, 4],
            'conv_list': [128, 256, 512, 1024],
            'samples': [2, 3, 4]
        }
        parameters_list.append(params)

        params = {
            'in_channel': 3 + 128 * 1,  # in_channel = lvconv_list[-1] * (n_layers + 1)
            'split_list': [3, 5, 5],
            'conv_list': [128, 256, 512, 1024],
            'samples': [2, 3, 4]
        }
        parameters_list.append(params)

        self.model = CubeCNN(num_class=self.num_class,
                             parameters_list=parameters_list,
                             normal_channel=args.normal).to(self.device)

        checkpoint = torch.load(str(savepath) + '/best_model.pth')
        # checkpoint = torch.load('D:/Workspace-PyCharm/CubeCNN/log/cls_CubeCNN/2026-10-10_14-55/checkpoints/best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(self.device)


    def value(self):
        def log_string(str_info):
            # self.logger.info(str_info)
            print(str_info)

        # self.logger.info('PARAMETER ...')
        # self.logger.info(self.args)

        log_string('Load dataset ...')
        DATA_PATH = 'data/modelnet40_normal_resampled/'

        TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=self.args.num_point, split='train',
                                           normal_channel=self.args.normal)
        TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=self.args.num_point, split='test',
                                          normal_channel=self.args.normal)

        trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=self.args.batch_size, shuffle=True,
                                                      num_workers=4)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=self.args.batch_size, shuffle=False,
                                                     num_workers=4)

        # 数据增强器
        augmenter = DataAugmentation(npoints=self.args.num_point, scale_low=0.9, scale_high=1.0, shift_range=0.1, normal=self.args.normal)

        test_loss_list = []
        best_test_results = {
            'best_OA': 0.0,
            'best_mAcc': 0.0,
            'best_mIoU': 0.0,
        }

        '''VALUING'''

        for epoch in range(self.args.epoch):
            log_string(f"Epoch {epoch + 1} ({epoch + 1}/{self.args.epoch}):")

            train_total_loss, train_results = self.test(trainDataLoader)

            test_total_loss, test_results = self.test(testDataLoader)
            test_loss_list.append(test_total_loss / len(testDataLoader))

            log_string(f"train_results: {train_results}")
            log_string(f"test_results: {test_results}")

            if test_results['OA'] >= best_test_results['best_OA']:
                best_test_results['best_OA'] = test_results['OA']

            if test_results['mAcc'] >= best_test_results['best_mAcc']:
                best_test_results['best_mAcc'] = test_results['mAcc']

            log_string(f"Instance_test_OA = {test_results['OA']}, Instance_test_mAcc = {test_results['mAcc']}")
            log_string(f"Best_test_OA = {best_test_results['best_OA']}, Best_test_mAcc = {best_test_results['best_mAcc']}")

        print('End of valuing...')

    def test(self, testDataLoader):
        self.model.eval()

        data_processer = DataAugmentation(npoints=self.args.num_point, scale_low=0.9, scale_high=1.0, shift_range=0.3,
                                     normal=self.args.normal)

        test_total_loss = 0
        test_metric = ClassificationMetric(num_classes=self.num_class, ignore_index=None)
        with torch.no_grad():
            for j, (points_batches, targets) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                print()

                print(f"Points_batches: {points_batches.shape}")
                # points_batches = points_batches.to(self.device)
                points_batches = data_processer.fps_sample_and_normalize(points_batches)
                # targets = targets[:, 0].long().to(self.device)
                targets = targets.long().to(self.device)

                root_fts, root_logits = self.model(points_batches)

                test_loss = self.criterion(root_logits, targets)
                test_total_loss += test_loss.item()

                pred = torch.argmax(root_logits, dim=-1)
                test_metric.update(pred, targets)

        test_results = test_metric.get_metrics()
        return test_total_loss, test_results


if __name__ == '__main__':
    print()
