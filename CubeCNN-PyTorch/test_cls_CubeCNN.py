import argparse
import sys
from pathlib import Path

# 文件所在目录（建议用绝对路径，避免工作目录影响）
module_dir = Path("./log/cls_CubeCNN/2026-10-15_15-14/").resolve()
if str(module_dir) not in sys.path:
    sys.path.insert(0, str(module_dir))

# 现在可以直接从模块名导入类、函数
from trainer_CubeCNN_cls_single import ModelTrainer



# from train_utils.tester_CubeCNN_cls_single import ModelTester

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('CubeCNN')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training [default: 24]')
    # parser.add_argument('--model', default='pointnet2_cls_msg', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=1, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer for training [default: Adam]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--scheduler', type=str, default='Cosine', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=True,
                        help='Whether to use normal information [default: False]')
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_dir = module_dir.joinpath('checkpoints/')
    trainer = ModelTrainer(args, checkpoint_dir, logger=None)

    trainer.value(model='last')


if __name__ == '__main__':
    main()
