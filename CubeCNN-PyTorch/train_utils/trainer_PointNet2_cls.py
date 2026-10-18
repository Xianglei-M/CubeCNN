import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import provider

'''
from data_utils.frameCN_ModelNetDataLoader import ModelNetDataset_3
from data_utils.data_augmentation import PointCloudAugment

from models.CubeCNN_cls import CubeCNN

from statistics_utils.segmentation_metric_util import SegmentationMetric
from statistics_utils.shapenet_partseg_statistics_util import ShapeNetIoU
'''

from pointnet_data_utils.ModelNetDataLoader import ModelNetDataLoader

from pointnet_models.pointnet2_cls_msg import PointNet2_msg, PointNet2_loss

from statistics_utils.segmentation_metric_util import ClassificationMetric

class PointNet2Trainer:
    def __init__(self, args, savepath, logger):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.args = args
        self.save_path = savepath
        self.logger = logger

        # self.augment = PointCloudAugment(args.normal)

        self.num_classes = 40
        self.model = PointNet2_msg(self.num_classes, normal_channel=args.normal).to(self.device)

        '''
        try:
            checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            log_string('Use pretrain model')
        except:
            log_string('No existing model, starting training from scratch...')
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

        # self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.criterion = PointNet2_loss().to(self.device)

    def train(self):
        def log_string(str):
            self.logger.info(str)
            print(str)

        '''HYPER PARAMETER'''
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        '''DATA LOADING'''
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

        '''MODEL LOADING'''
        # num_class = 40
        # MODEL = importlib.import_module(args.model)

        global_epoch = 0
        global_step = 0
        best_instance_acc = 0.0
        best_class_acc = 0.0
        mean_correct = []

        '''TRANING'''
        self.logger.info('Start training...')
        # for epoch in range(start_epoch, self.args.epoch):
        for epoch in range(self.args.epoch):
            log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, self.args.epoch))

            train_metric = ClassificationMetric(num_classes=self.num_classes, ignore_index=None)

            self.scheduler.step()
            for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
                print()

                points, target = data
                points = points.data.numpy()
                points = provider.random_point_dropout(points)
                points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
                points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
                points = torch.Tensor(points)
                target = target[:, 0]

                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                self.optimizer.zero_grad()

                self.model = self.model.train()
                pred, trans_feat = self.model(points)
                loss = self.criterion(pred, target.long(), trans_feat)
                pred_choice = pred.data.max(1)[1]

                train_metric.update(pred_choice, target)

                correct = pred_choice.eq(target.long().data).cpu().sum()
                mean_correct.append(correct.item() / float(points.size()[0]))
                loss.backward()
                self.optimizer.step()
                global_step += 1

            train_instance_acc = np.mean(mean_correct)
            log_string('Train Instance Accuracy: %f' % train_instance_acc)

            train_results = train_metric.get_metrics()
            log_string(f"train_results: {train_results}")

            with torch.no_grad():
                instance_acc, class_acc, test_results = self.test(self.model.eval(), testDataLoader)

                log_string(f"test_results: {test_results}")

                if (instance_acc >= best_instance_acc):
                    best_instance_acc = instance_acc
                    best_epoch = epoch + 1

                if (class_acc >= best_class_acc):
                    best_class_acc = class_acc
                log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
                log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

                if (instance_acc >= best_instance_acc):
                    self.logger.info('Save model...')
                    # savepath = str(checkpoints_dir) + '/best_model.pth'
                    log_string('Saving at %s' % self.save_path)
                    state = {
                        'epoch': best_epoch,
                        'instance_acc': instance_acc,
                        'class_acc': class_acc,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }
                    torch.save(state, self.save_path)
                global_epoch += 1

        self.logger.info('End of training...')

    def test(self, model, loader, num_class=40):
        mean_correct = []
        class_acc = np.zeros((self.num_classes, 3))

        test_metric = ClassificationMetric(num_classes=self.num_classes, ignore_index=None)

        for j, data in tqdm(enumerate(loader), total=len(loader)):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = model.eval()
            pred, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]

            test_metric.update(pred_choice, target)

            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
                class_acc[cat, 1] += 1
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
        class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
        class_acc = np.mean(class_acc[:, 2])
        instance_acc = np.mean(mean_correct)

        test_results = test_metric.get_metrics()

        return instance_acc, class_acc, test_results
    '''
    def train_2(self):
        def log_string(str_info):
            self.logger.info(str_info)
            print(str_info)

        self.logger.info('PARAMETER ...')
        self.logger.info(self.args)

        DATA_PATH = './data/modelnet40_normal_resampled/'
        TRAIN_DATASET = ModelNetDataset_3(root=DATA_PATH, data_set='train', normal_channel=self.args.normal,
                                          cache_size=15000)
        trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=self.args.batch_size, shuffle=True)

        TEST_DATASET = ModelNetDataset_3(root=DATA_PATH, data_set='test', normal_channel=self.args.normal,
                                         cache_size=15000)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=self.args.batch_size, shuffle=True)

        train_loss_list = []
        test_loss_list = []
        best_test_OA = 0.0
        best_test_mAcc = 0.0

        """TRANING"""
        self.logger.info('Start training...')
        self.model.train()

        # for epoch in range(start_epoch, args.epoch):
        for epoch in range(self.args.epoch):
            log_string(f"Epoch {epoch + 1} ({epoch + 1}/{self.args.epoch}):")

            self.model.train()

            self.scheduler.step()

            # augment = PointCloudAugment(self.args.normal)
            train_metric = SegmentationMetric(num_classes=self.num_classes, ignore_index=None)

            total_loss = 0
            for batch_id, (points_batches, targets) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader),
                                                            smoothing=0.9):
                print()

                targets = targets.to(self.device).long()

                points_batches = points_batches.numpy()
                points_batches = self.augment(points_batches)
                points_batches = self.augment.pc_normalize(points_batches)

                self.optimizer.zero_grad()
                root_fts, root_logits = self.model(points_batches)

                loss = self.criterion(root_logits, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pred = torch.argmax(root_logits, dim=-1)
                train_metric.update(pred, targets)

            train_loss_list.append(total_loss / len(trainDataLoader))
            train_results = train_metric.get_metrics()
            log_string(f"train_results: {train_results}")

            test_total_loss, test_results = self.test(testDataLoader)
            test_loss_list.append(test_total_loss / len(testDataLoader))
            log_string(f"test_results: {test_results}")

            if test_results['OA'] >= best_test_OA:
                best_test_OA = test_results['OA']

            if test_results['mAcc'] >= best_test_mAcc:
                best_test_mAcc = test_results['mAcc']

            log_string(f"Instance_test_OA: {test_results['OA']}, Instance_test_mAcc Accuracy: {test_results['mAcc']}")
            log_string(f"Best_test_OA: {best_test_OA}, Best_test_mAcc Accuracy: {best_test_mAcc}")

            if test_results['OA'] >= best_test_OA:
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
        self.logger.info('End of training...')

    def test_2(self, testDataLoader):
        self.model.eval()

        test_total_loss = 0
        test_metric = SegmentationMetric(num_classes=self.num_classes, ignore_index=None)
        with torch.no_grad():
            for j, (points_batches, targets) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                print()

                targets = targets.to(self.device).long()

                points_batches = points_batches.numpy()
                # points_batches = self.augment(points_batches)
                points_batches = self.augment.pc_normalize(points_batches)

                # self.optimizer.zero_grad()

                root_fts, root_logits = self.model(points_batches)

                test_loss = self.criterion(root_logits, targets)
                test_total_loss += test_loss.item()

                pred = torch.argmax(root_logits, dim=-1)
                test_metric.update(pred, targets)

        test_results = test_metric.get_metrics()
        return test_total_loss, test_results
    '''


if __name__ == '__main__':
    print()
