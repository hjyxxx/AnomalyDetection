import argparse

import torch.random
from cprint import cprint

from exp.exp_pr import ExpPR
from exp.exp_pre import ExpPre
from exp.exp_rec import ExpRec
from exp.exp_ws import ExpWS
from utils.util import load_config, load_args

torch.random.manual_seed(2024)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # 基础参数
    parser.add_argument('--is-training', type=bool, default=True, help='训练或者测试')
    parser.add_argument('--task-name', type=str, default='rec', help='任务类型')
    parser.add_argument('--debug', type=bool, default=False, help="debug")
    parser.add_argument('--model', type=str, default='Transformer', help='模型名称')
    parser.add_argument('--train-epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--use-gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--gpu', type=int, default=0, help='使用第几个GPU')

    # 模型参数
    parser.add_argument('--e-layers', type=int, default=1, help='')
    parser.add_argument('--in-features', type=int, default=2, help='输入维度')
    parser.add_argument('--out-features', type=int, default=2, help='输出维度')
    parser.add_argument('--embedding-channels', type=int, default=64, help="嵌入维度")
    parser.add_argument('--d-ff', type=int, default=128, help='')
    parser.add_argument('--patch_len', type=int, default=6, help="")
    parser.add_argument('--stride', type=int, default=2, help="")
    parser.add_argument('--padding', type=int, default=2, help="")
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--fusion', type=str, default='t', help="时域/频域/小波域, tfw")

    # 数据集参数
    parser.add_argument('--data', type=str, default='asd', help='数据集名称')
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

    if args.is_training:
        args = load_config(args, parser)
    else:
        folder_path = 'checkpoints/pre/PureGraph/shtc/train/save0'
        args = load_args(folder_path)

    cprint.info(args)

    if args.task_name == 'rec':
        Exp = ExpRec
    elif args.task_name == 'pre':
        Exp = ExpPre
    elif args.task_name == 'pr':
        Exp = ExpPR
    elif args.task_name == 'ws':
        Exp = ExpWS

    else:
        raise ValueError("Do Not Exist This Value: {}".format(args.task_name))

    exp = Exp(args)
    if args.is_training:
        exp.train()
    else:
        exp.test()

