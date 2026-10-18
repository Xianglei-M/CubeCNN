import torch
from torch import nn
from tqdm import tqdm
import provider

from data_utils.ShapeNetDataLoader import PartNormalDataset

from models.CubeCNN_seg import CubeCNN_PartSeg

from statistics_utils.segmentation_metric_util import ClassificationMetric
from statistics_utils.shapenet_partseg_statistics_util import ShapeNetIoU


class ModelTrainer:
    def __init__(self, args, savepath, logger):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.args = args
        self.save_path = savepath
        self.logger = logger

        # self.augment = PointCloudAugment(args.normal)

        self.num_class = 16
        self.num_part = 50
        self.model = CubeCNN_PartSeg(num_class=self.num_class, num_part=self.num_part, split_list=[5, 3, 3], cbconv_list=[128, 256, 512],
                                     samples_list=[[2, 3, 4], [2, 3, 4], [3, 5, 7], [5, 7, 9]],
                                     lvconv_list=[32, 64, 128, 256],
                                     normal_channel=args.normal).to(self.device)

        '''
        try:
            checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
            start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # log_string('Use pretrain model')
            print('Use pretrain model')
        except:
            # log_string('No existing model, starting training from scratch...')
            print('No existing model, starting training from scratch...')
            start_epoch = 0
        '''

        if args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=args.decay_rate
            )
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.7)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(self.device)

    def data_augmentation(self, points):
        points = points.numpy()

        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])

        points = provider.normalize_data(points)

        points = torch.from_numpy(points).to(self.device)
        return points

    def train(self):
        def log_string(str_info):
            self.logger.info(str_info)
            print(str_info)

        self.logger.info('PARAMETER ...')
        self.logger.info(self.args)

        log_string('Load dataset ...')
        root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

        TRAIN_DATASET = PartNormalDataset(root=root, npoints=self.args.num_point, split='trainval', normal_channel=self.args.normal)
        trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=self.args.batch_size, shuffle=True,
                                                      num_workers=8)
        TEST_DATASET = PartNormalDataset(root=root, npoints=self.args.num_point, split='test', normal_channel=self.args.normal)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=self.args.batch_size, shuffle=False,
                                                     num_workers=8)
        log_string("The number of training data is: %d" % len(TRAIN_DATASET))
        log_string("The number of test data is: %d" % len(TEST_DATASET))

        train_loss_list = []
        test_loss_list = []
        best_test_OA = 0.0
        best_test_mAcc = 0.0

        '''TRANING'''
        self.logger.info('Start training...')

        for epoch in range(self.args.epoch):
            log_string(f"Epoch {epoch + 1} ({epoch + 1}/{self.args.epoch}):")

            self.model.train()

            # self.scheduler.step()

            train_metric = ClassificationMetric(num_classes=self.num_part, ignore_index=None)
            train_iou = ShapeNetIoU(num_classes=self.num_part)

            total_loss = 0
            for batch_id, (points_batches, classes, targets) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader),
                                                            smoothing=0.9):
                print()

                print(f"Points_batches: {points_batches.shape}")
                '''
                points_batches = points_batches.numpy()
                points_batches = self.augment(points_batches)
                points_batches = self.augment.pc_normalize(points_batches)
                '''
                points_batches = self.data_augmentation(points_batches)
                targets = targets.long().to(self.device)
                # targets = targets[:, 0].long().to(self.device)

                self.optimizer.zero_grad()
                leaves_fts, leaves_logits = self.model(points_batches)

                pred = torch.argmax(leaves_logits, dim=-1)
                train_metric.update(pred, targets)
                train_iou.update(pred, targets)

                leaves_logits = leaves_logits.view(-1, self.num_part)
                targets = targets.view(-1)
                loss = self.criterion(leaves_logits, targets)
                loss.backward()
                self.optimizer.step()

                # total_loss += loss.item()

            # train_loss_list.append(total_loss / len(trainDataLoader))
            train_results = train_metric.get_metrics()
            train_iou_res = train_iou.get_statistics()
            log_string(f"train_results: {train_results}")
            log_string(f"train_iou_results: {train_iou_res}")

            test_total_loss, test_results, test_iou_res = self.test(testDataLoader)
            test_loss_list.append(test_total_loss / len(testDataLoader))
            log_string(f"test_results: {test_results}")
            log_string(f"test_iou_results: {test_iou_res}")

            if test_results['mAcc'] >= best_test_mAcc:
                best_test_mAcc = test_results['mAcc']

            # if test_results['OA'] >= best_test_OA:
            if test_results['OA'] >= best_test_OA:
                best_test_OA = test_results['OA']

                best_epoch = epoch + 1
                self.logger.info('Save model...')
                # savepath = str(self.checkpoints_dir) + '/best_model.pth'
                savepath = self.save_path
                log_string('Saved at %s' % savepath)

                state = {
                    'epoch': best_epoch,
                    'OA': test_results['OA'],
                    'mAcc': test_results['mAcc'],
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                torch.save(state, savepath)

            log_string(f"Instance_test_OA: {test_results['OA']}, Instance_test_mAcc Accuracy: {test_results['mAcc']}")
            log_string(f"Best_test_OA: {best_test_OA}, Best_test_mAcc Accuracy: {best_test_mAcc}")

            self.scheduler.step()
        self.logger.info('End of training...')

    def test(self, testDataLoader):
        self.model.eval()

        test_total_loss = 0
        test_metric = ClassificationMetric(num_classes=self.num_part, ignore_index=None)
        test_iou = ShapeNetIoU(num_classes=self.num_part)
        with torch.no_grad():
            for j, (points_batches, classes, targets) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                print()

                print(f"Points_batches: {points_batches.shape}")
                points_batches = points_batches.to(self.device)
                # targets = targets[:, 0].long().to(self.device)
                targets = targets.long().to(self.device)

                leaves_fts, leaves_logits = self.model(points_batches)

                pred = torch.argmax(leaves_logits, dim=-1)
                test_metric.update(pred, targets)
                test_iou.update(pred, targets)

                leaves_logits = leaves_logits.view(-1, self.num_part)
                targets = targets.view(-1)
                test_loss = self.criterion(leaves_logits, targets)
                test_total_loss += test_loss.item()

        test_results = test_metric.get_metrics()
        test_iou_res = test_iou.get_statistics()
        return test_total_loss, test_results, test_iou_res


if __name__ == '__main__':
    print()
