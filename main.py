import argparse

from cprint import cprint

from exp.exp_rec import ExpRec
from utils.util import load_data_config

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # 基础参数
    parser.add_argument('--is-training', type=bool, default=True, help='训练或者测试')
    parser.add_argument('--task-name', type=str, default='rec', help='任务类型')
    parser.add_argument('--debug', type=bool, default=True, help="debug")
    parser.add_argument('--model', type=str, default='test', help='模型名称')
    parser.add_argument('--train-epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--use-gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--gpu', type=int, default=0, help='使用第几个GPU')

    # 数据集参数
    parser.add_argument('--data', type=str, default='shtc', help='数据集名称')
    parser.add_argument('--data-path', type=str, default='pose/', help='数据集路径')
    parser.add_argument('--data-config', type=str, default='configs/', help="数据集配置文件路径")

    # DataLoader参数
    parser.add_argument('--batch-size', type=int, default=128, help='批次大小')
    parser.add_argument('--num-workers', type=int, default=4, help='')
    parser.add_argument('--pin-memory', type=bool, default=True, help='')

    # optimizer参数
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')

    # 日志
    parser.add_argument('--checkpoints', type=str, default='checkpoints', help='日志路径')

    args = parser.parse_args()

    args = load_data_config(args, parser)

    if args.task_name == 'rec':
        Exp = ExpRec

    else:
        raise ValueError("Do Not Exist This Value: {}".format(args.task_name))

    if args.is_training:
        exp = Exp(args)
        cprint.info(args)
        exp.train()

