import argparse
import numpy as np
import os
import torch
from torch import nn
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil

from models_7.frameConvNet_partseg_2 import FrameConvNet_PartSeg

from statistics_utils.segmentation_metric_util import ClassificationMetric
from statistics_utils.shapenet_partseg_statistics_util import ShapeNetIoU

'''
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
'''

"""
需要配置的参数：
--model pointnet2_cls_msg 
--normal 
--log_dir pointnet2_cls_msg
"""


class ModelTrainer:
    def __init__(self, args, checkpoints_dir, logger):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = model.to(self.device)
        self.num_classes = 50
        self.model = FrameConvNet_PartSeg(num_classes=self.num_classes, split_list=[3, 2, 3],
                                          mlp_list=[[128, 256, 256], [128, 128, 256], [128, 128, 256], [128, 128, 256]],
                                          fp_list=[[128, 128, 256], [128, 128, 256], [128, 128, 256]],
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
        self.checkpoints_dir = checkpoints_dir
        self.logger = logger

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

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

    def train(self, trainDataLoader, testDataLoader):
        def log_string(str_info):
            self.logger.info(str_info)
            print(str_info)

        # shutil.copy(f'./models_2/frameConv_modules.py', str(experiment_dir))
        # shutil.copy('./models_2/frameConv_utils.py', str(experiment_dir))

        self.logger.info('PARAMETER ...')
        self.logger.info(self.args)

        global_epoch = 0
        global_step = 0
        best_instance_acc = 0.0
        best_class_acc = 0.0
        mean_correct = []

        train_best_OA = 0
        test_best_OA = 0
        '''TRANING'''
        self.logger.info('Start training...')

        # for epoch in range(start_epoch, args.epoch):
        for epoch in range(self.args.epoch):
            log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, self.args.epoch))

            self.scheduler.step()
            train_metric = ClassificationMetric(num_classes=self.num_classes, ignore_index=-1)
            total_loss = 0
            for batch_id, (data_batches, shape_cls) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader),
                                                          smoothing=0.9):
                print()
                # points, targets = data

                data_batches = data_batches.numpy()
                data_batches = provider.random_point_dropout(data_batches)
                data_batches[:, :, 0:3] = provider.random_scale_point_cloud(data_batches[:, :, 0:3])
                data_batches[:, :, 0:3] = provider.shift_point_cloud(data_batches[:, :, 0:3])
                # points = torch.Tensor(points)
                data_batches = torch.from_numpy(data_batches)
                print(f"data_batches: {data_batches.shape}")
                print(f"shape_cls: {shape_cls.shape}")
                # targets = targets[:, 0]
                # print(f"new_targets: {targets.shape}")

                data_batches, shape_cls = data_batches.to(self.device), shape_cls.to(self.device).long()

                self.optimizer.zero_grad()
                pts_layers, fts_layers, plb_layers = self.model.frameConv_data_process_train(data_batches)
                leaves_logits, conv_fts_layers = self.model(pts_layers, fts_layers)
                print(f"leaves_logits: {leaves_logits.shape}")

                leaves_logits = leaves_logits.reshape(-1, self.num_classes)
                part_labels = torch.tensor(plb_layers[0]).reshape(-1).to(self.device).long()
                print(f"leaves_logits: {leaves_logits.shape}")
                print(f"part_labels: {part_labels.shape}")

                loss = self.criterion(leaves_logits, part_labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                global_step += 1

                pred = torch.argmax(leaves_logits, dim=-1)
                leaves_pts = torch.tensor(pts_layers[0]).reshape(-1, 3).to(self.device)
                mask = (leaves_pts.abs().sum(dim=-1) == 0)
                print(f"mask: {mask.shape}")
                pred[mask] = -1
                print(f"pred: {pred.shape}")

                # for i in range(len(part_labels)):
                #     if (pred[i] == -1) or (part_labels[i] == -1):
                #         if pred[i] != part_labels[i]:
                #             print(f"pred[i]  part_labels[i]，不同时等于-1")
                #     continue

                train_metric.update(pred, part_labels)

            train_results = train_metric.get_metrics()
            log_string(f"train_results: {train_results}")

            test_results = self.test(testDataLoader)
            log_string(f"test_results: {test_results}")

            if test_results['OA'] >= best_instance_acc:
                best_instance_acc = test_results['OA']
                best_epoch = epoch + 1

            if test_results['mAcc'] >= best_class_acc:
                best_class_acc = test_results['mAcc']
            log_string(
                'Test Instance Accuracy: %f, Class Accuracy: %f' % (test_results['OA'], test_results['mAcc']))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if test_results['OA'] >= best_instance_acc:
                self.logger.info('Save model...')
                savepath = str(self.checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)

                state = {
                    'epoch': best_epoch,
                    'instance_acc': test_results['OA'],
                    'class_acc': test_results['mAcc'],
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

        self.logger.info('End of training...')

    def test(self, testDataLoader):
        self.model.eval()
        mean_correct = []
        class_acc = np.zeros((self.num_classes, 3))

        test_metric = ClassificationMetric(num_classes=self.num_classes, ignore_index=-1)
        with torch.no_grad():
            for j, (data_batches, shape_cls) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                data_batches, shape_cls = data_batches.to(self.device), shape_cls.to(self.device).long()

                self.optimizer.zero_grad()
                pts_layers, fts_layers, plb_layers = self.model.frameConv_data_process_train(data_batches)
                leaves_logits, conv_fts_layers = self.model(pts_layers, fts_layers)
                print(f"root_logits: {leaves_logits.shape}")

                leaves_logits = leaves_logits.reshape(-1, self.num_classes)
                part_labels = torch.tensor(plb_layers[0]).reshape(-1).long()
                pred = torch.argmax(leaves_logits, dim=-1)

                leaves_pts = torch.tensor(pts_layers[0]).reshape(-1, 3).to(self.device)
                mask = (leaves_pts.abs().sum(dim=-1) == 0)
                print(f"mask: {mask.shape}")
                pred[mask] = -1
                print(f"pred: {pred.shape}")

                test_metric.update(pred, part_labels)

        test_results = test_metric.get_metrics()
        return test_results


if __name__ == '__main__':
    print()
