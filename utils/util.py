import argparse
import json
import os
from collections import deque

import numpy as np
import yaml


def check_config_cache(path, yaml_data):
    """

    :param path:
    :return:
    """
    cache_path = os.path.join(path, 'config_cache.yaml')
    cache_flag = False
    # cache不存在, 创建cache
    if not os.path.exists(cache_path):
        with open(cache_path, 'w') as f:
            yaml.safe_dump(yaml_data, f)
    # cache存在, 对比文件内容是否相同
    else:
        with open(cache_path, 'r') as f:
            cache_data = yaml.safe_load(f)
        if cache_data == yaml_data:
            cache_flag = True
        else:
            with open(cache_path, 'w') as f:
                yaml.safe_dump(yaml_data, f)

    return cache_flag


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


def bfs_traversal(dictionary):
    queue = deque([(dictionary, '')])  # 初始化队列，将字典和当前路径传入
    res = []
    while queue:
        node, path = queue.popleft()  # 出队列
        if isinstance(node, dict):  # 如果当前节点是字典
            for key, value in node.items():
                new_path = f"{path}.{key}" if path else key  # 更新路径
                # print(new_path, ":", value)  # 输出路径和值
                res.append([new_path, value])
                queue.append((value, new_path))  # 将子节点和路径加入队列

    res = [item for item in res if not isinstance(item[1], dict)]

    return res


def get_iter_num(res):
    """
    得到遍历层数
    :param res:
    :return:
    """
    max_row = 1
    for item in res:
        if len(item[0].split('.')) > max_row:
            max_row = len(item[0].split('.'))

    return max_row + 1


def change_dict_to_str(dictionary):
    # 广度优先遍历字典
    res = bfs_traversal(dictionary)

    # 得到行数
    iter_num = get_iter_num(res)

    tplt = ""
    for i in range(iter_num):
        row_s = []
        for item in res:
            keys = item[0]
            value = item[1]
            keys = keys.split('.')
            if i < len(keys):
                key = keys[i]
            elif i == len(keys):
                key = value
            else:
                key = " "
            row_s.append(key)

        for j in range(len(row_s)):
            if isinstance(row_s[j], int):
                tplt += "|{:^10d}|\t"
            elif isinstance(row_s[j], np.float32) or isinstance(row_s[j], np.float64) or isinstance(row_s[j], float):
                tplt += "|{:^10.6f}|\t"
            else:
                tplt += "|{:^10s}|\t"
        tplt = tplt.format(*row_s)
        tplt = tplt.strip()
        tplt += "\n"

    tplt = tplt.strip()

    return tplt
