import torch
from torch import nn
from tqdm import tqdm
# import provider

from models.CubeCNN_cls import CubeCNN

from data_utils.old_augmentation import PointCloudAugment

from statistics_utils.segmentation_metric_util import ClassificationMetric
from statistics_utils.shapenet_partseg_statistics_util import ShapeNetIoU


class ModelTrainer:
    def __init__(self, args, savepath, logger):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = model.to(self.device)
        self.num_classes = 40
        # self.model = FrameConvNet_5(num_classes=self.num_classes, split_list=[3, 3, 3],
        #                             mlp_list=[[64, 64, 128], [128, 128, 256], [256, 256, 512], [512, 512, 1024]],
        #                             normal_channel=args.normal).to(self.device)
        self.model = CubeCNN(num_classes=self.num_classes, split_list=[3, 3, 3],
                             lvconv_list=[[16, 32, 16], [32, 64, 32], [256, 512, 256], [512, 1024, 512]],
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

        self.args = args
        self.save_path = savepath
        self.logger = logger

        self.augment = PointCloudAugment(args.normal)

        if args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=args.learning_rate,
                # betas=(0.9, 0.999),
                # eps=1e-08,
                weight_decay=args.decay_rate
            )
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.7)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def train_2(self, dataloader, epochs):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

    '''
    def data_augmentation(self, data_batches):
        data_batches = data_batches.numpy()

        data_batches = provider.random_point_dropout(data_batches)
        data_batches[:, :, 0:3] = provider.random_scale_point_cloud(data_batches[:, :, 0:3])
        data_batches[:, :, 0:3] = provider.shift_point_cloud(data_batches[:, :, 0:3])

        data_batches = torch.from_numpy(data_batches)
        return data_batches
    '''

    def train(self, trainDataLoader, testDataLoader):
        def log_string(str_info):
            self.logger.info(str_info)
            print(str_info)

        self.logger.info('PARAMETER ...')
        self.logger.info(self.args)

        train_loss_list = []
        test_loss_list = []
        best_epoch = 0
        best_test_OA = 0.0
        best_test_mAcc = 0.0


        '''TRANING'''
        self.logger.info('Start training...')
        # for epoch in range(start_epoch, args.epoch):
        for epoch in range(self.args.epoch):
            log_string(f"Epoch {epoch + 1} ({epoch + 1}/{self.args.epoch}):")

            self.scheduler.step()

            # augment = PointCloudAugment(self.args.normal)
            train_metric = ClassificationMetric(num_classes=self.num_classes, ignore_index=None)

            total_loss = 0
            for batch_id, (data_batches, targets) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader),
                                                          smoothing=0.9):
                print()

                '''data_batches = self.data_augmentation(data_batches)'''

                data_batches = self.augment(data_batches.numpy())

                data_batches = self.augment.pc_normalize(data_batches)

                data_batches = torch.tensor(
                    data_batches,
                    dtype=torch.float32
                ).to(self.device)

                # data_batches, targets = data_batches.to(self.device), targets.to(self.device).long()
                targets = targets.to(self.device).long()

                self.optimizer.zero_grad()
                cen_list, psn_list, pts_layers, norm_layers = self.model.frameConv_data_process_train(data_batches)
                root_logits, conv_fts_layers = self.model(cen_list, psn_list, pts_layers, norm_layers)
                # print(f"root_logits: {root_logits.shape}")

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

    def test(self, testDataLoader):
        self.model.eval()

        test_total_loss = 0
        test_metric = ClassificationMetric(num_classes=self.num_classes, ignore_index=None)
        with torch.no_grad():
            for j, (data_batches, targets) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                print()

                data_batches, targets = data_batches.to(self.device), targets.to(self.device).long()

                self.optimizer.zero_grad()
                cen_list, psn_list, pts_layers, norm_layers = self.model.frameConv_data_process_train(data_batches)
                root_logits, conv_fts_layers = self.model(cen_list, psn_list, pts_layers, norm_layers)
                # print(f"root_logits: {root_logits.shape}")
                test_loss = self.criterion(root_logits, targets)
                test_total_loss += test_loss.item()

                pred = torch.argmax(root_logits, dim=-1)
                test_metric.update(pred, targets)

        test_results = test_metric.get_metrics()
        return test_total_loss, test_results


if __name__ == '__main__':
    print()
