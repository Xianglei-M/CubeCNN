import torch
from torch import nn
from tqdm import tqdm
import provider

from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.data_augmentation import DataAugmentation

from models.CubeCNN_cls import CubeCNN

from statistics_utils.segmentation_metric_util import ClassificationMetric
from statistics_utils.shapenet_partseg_statistics_util import ShapeNetIoU


class ModelTrainer:
    def __init__(self, args, savepath, logger):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.args = args
        self.save_path = savepath
        self.logger = logger

        # self.augment = PointCloudAugment(args.normal)

        self.num_classes = 10
        self.model = CubeCNN(num_classes=self.num_classes, split_list=[5, 5, 3], cbconv_list=[256, 512, 1024],
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

        # optimizer
        if args.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=args.decay_rate
            )
        elif args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=args.decay_rate
            )
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        # scheduler
        if args.scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=args.epoch,
                eta_min=1e-5
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.7)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(self.device)

    def data_augmentation(self, points):
        points = points.numpy()

        points = provider.random_point_dropout(points)
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
        DATA_PATH = 'data/modelnet40_normal_resampled/'

        TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=self.args.num_point, split='train',
                                           normal_channel=self.args.normal)
        TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=self.args.num_point, split='test',
                                          normal_channel=self.args.normal)

        trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=self.args.batch_size, shuffle=True,
                                                      num_workers=8)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=self.args.batch_size, shuffle=False,
                                                     num_workers=8)

        # 数据增强器
        augmenter = DataAugmentation(npoints=self.args.num_point, scale_low=0.9, scale_high=1.0, shift_range=0.1, normal=self.args.normal)

        train_loss_list = []
        test_loss_list = []
        best_test_OA = 0.0
        best_test_mAcc = 0.0

        '''TRANING'''
        self.logger.info('Start training...')

        for epoch in range(self.args.epoch):
            log_string(f"Epoch {epoch + 1} ({epoch + 1}/{self.args.epoch}):")

            self.model.train()

            train_metric = ClassificationMetric(num_classes=self.num_classes, ignore_index=None)

            total_loss = 0
            for batch_id, (points_batches, targets) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader),
                                                            smoothing=0.9):
                print()
                '''
                points_batches = points_batches.numpy()
                points_batches = self.augment(points_batches)
                points_batches = self.augment.pc_normalize(points_batches)
                '''

                print(f"Points_batches: {points_batches.shape}")
                points_batches = augmenter(points_batches)
                targets = targets.long().to(self.device)
                # points_batches = self.data_augmentation(points_batches)
                # targets = targets[:, 0].long().to(self.device)

                self.optimizer.zero_grad()
                root_fts, root_logits = self.model(points_batches)

                loss = self.criterion(root_logits, targets)
                loss.backward()
                self.optimizer.step()

                # total_loss += loss.item()
                pred = torch.argmax(root_logits, dim=-1)
                train_metric.update(pred, targets)

            # train_loss_list.append(total_loss / len(trainDataLoader))
            train_results = train_metric.get_metrics()
            log_string(f"train_results: {train_results}")

            test_total_loss, test_results = self.test(testDataLoader)
            test_loss_list.append(test_total_loss / len(testDataLoader))
            log_string(f"test_results: {test_results}")

            if test_results['mAcc'] >= best_test_mAcc:
                best_test_mAcc = test_results['mAcc']

            log_string(f"Instance_test_OA: {test_results['OA']}, Instance_test_mAcc Accuracy: {test_results['mAcc']}")
            log_string(f"Best_test_OA: {best_test_OA}, Best_test_mAcc Accuracy: {best_test_mAcc}")

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

            self.scheduler.step()
        self.logger.info('End of training...')

    def test(self, testDataLoader):
        self.model.eval()

        data_processer = DataAugmentation(npoints=self.args.num_point, scale_low=0.9, scale_high=1.0, shift_range=0.3,
                                     normal=self.args.normal)

        test_total_loss = 0
        test_metric = ClassificationMetric(num_classes=self.num_classes, ignore_index=None)
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
