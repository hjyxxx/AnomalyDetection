import os

import yaml


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


def load_data_config(args, parser):
    """
    加载配置文件
    :param args:
    :param parser:
    :return:
    """
    config_path = os.path.join(args.data_config, f'{args.data}.yaml')
    yaml_data = load_yaml(config_path)
    add_yaml_to_argparse(yaml_data, parser)
    args = parser.parse_args()

    if args.ckp:
        args.node_num = len(args.ckp)

    return args
