import argparse
import datetime
import logging
from pathlib import Path

import shutil

from train_utils.trainer_CubeCNN_cls_single import ModelTrainer


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('CubeCNN')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training [default: 24]')
    # parser.add_argument('--model', default='pointnet2_cls_msg', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer for training [default: Adam]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--scheduler', type=str, default='Cosine', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=True,
                        help='Whether to use normal information [default: False]')
    return parser.parse_args()


def create_dir(args):
    """CREATE DIR"""
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    # experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir = experiment_dir.joinpath('cls_CubeCNN/')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    return experiment_dir, checkpoints_dir, log_dir


def create_logger(log_dir):
    """LOG"""
    # args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler = logging.FileHandler(f'{log_dir}/classify_train_log.txt')
    # file_handler = logging.FileHandler(f'{log_dir}/{args.model}.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # log_string('PARAMETER ...')
    # log_string(args)
    return logger


def main():
    args = parse_args()

    experiment_dir, checkpoints_dir, log_dir = create_dir(args)
    logger = create_logger(log_dir)

    shutil.copy('train_utils/trainer_CubeCNN_cls_single.py', str(experiment_dir))
    shutil.copy('models/CubeCNN_cls.py', str(experiment_dir))
    shutil.copy('./models/CubeConv_modules.py', str(experiment_dir))
    shutil.copy('./models/CubeConv_utils.py', str(experiment_dir))

    savepath = str(checkpoints_dir) + '/best_model.pth'
    trainer = ModelTrainer(args, checkpoints_dir, logger=logger)

    trainer.train()


if __name__ == '__main__':
    main()
