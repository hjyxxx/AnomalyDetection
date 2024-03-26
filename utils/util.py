import argparse
import json
import os

import yaml
from cprint import cprint

from utils.logging import create_folder


def check_config_cash(path, yaml_data):
    """

    :param path:
    :return:
    """
    cash_path = os.path.join(path, 'config_cash.yaml')
    cash_flag = False
    # cash不存在, 创建cash
    if not os.path.exists(cash_path):
        with open(cash_path, 'w') as f:
            yaml.safe_dump(yaml_data, f)
    # cash存在, 对比文件内容是否相同
    else:
        with open(cash_path, 'r') as f:
            cash_data = yaml.safe_load(f)
        if cash_data == yaml_data:
            cash_flag = True
        else:
            with open(cash_path, 'w') as f:
                yaml.safe_dump(yaml_data, f)

    return cash_flag


def load_yaml(file_path):
    """
    加载yaml文件数据
    :param file_path: 文件路径
    :return:
    """
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def add_yaml_to_argparse(yaml_data, parser):
    """
    将配置文件中的数据加入到argparse中
    :param yaml_data:
    :param parser:
    :return:
    """
    for key, value in yaml_data.items():
        if isinstance(value, dict):
            subparser = parser.add_argument_group(key)
            add_yaml_to_argparse(value, subparser)
        else:
            parser.add_argument(f"--{key}", default=value)


def load_config(args, parser):
    """
    加载配置文件
    :param args:
    :param parser:
    :return:
    """

    # 创建日志文件夹
    logging_folder = create_folder(args)

    config_path = os.path.join(args.data_config, args.task_name, f'{args.data}.yaml')
    cash_path = os.path.join('checkpoints', args.task_name, args.model, args.data)
    yaml_data = load_yaml(config_path)
    if args.is_training and not args.debug:
        cash_flag = check_config_cash(cash_path, yaml_data)
    else:
        cash_flag = False
    add_yaml_to_argparse(yaml_data, parser)

    parser.add_argument(f"--folder_path", default=logging_folder)
    parser.add_argument(f"--cash_path", default=cash_path)
    parser.add_argument(f"--cash_flag", default=cash_flag)
    args = parser.parse_args()

    if args.ckp:
        args.node_num = len(args.ckp)

    return args


def load_args(folder_path):
    """
    加载参数
    :param folder_path:
    :return:
    """
    arg_path = os.path.join(folder_path, 'args.json')

    if os.path.exists(arg_path):

        with open(arg_path, 'r') as f:
            json_data = json.load(f)

        parser = argparse.ArgumentParser()
        add_yaml_to_argparse(json_data, parser)

        # parser.add_argument(f"--folder_path", default=folder_path)
        parser.set_defaults(is_training=False)

        args = parser.parse_args()

    else:
        raise ValueError("Do Not Exists This Value: {}".format(arg_path))

    return args
