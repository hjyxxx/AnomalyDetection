import json
import os

import torch
from cprint import cprint


def save_args(folder_path, args):
    """
    保存参数
    :return:
    """
    args_path = os.path.join(folder_path, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f)


def save_model(folder_path, model_info):
    """
    保存模型结构
    :param folder_path:
    :param model_info:
    :return:
    """
    path = os.path.join(folder_path, 'model.txt')
    with open(path, 'w') as f:
        f.write(model_info)


def save_results(folder_path, train_loss, vali_loss, test_loss, lr, results, epoch):
    """
    :param folder_path:
    :param train_loss:
    :param vali_loss:
    :param test_loss:
    :param lr:
    :param results:
    :param epoch:
    :return:
    """
    vali_loss = vali_loss if vali_loss is not None else 0
    test_loss = test_loss if test_loss is not None else 0

    basic_info = f"| Epoch {epoch:02d} | lr: {lr:02.6f} | train_loss {train_loss:5.8f} | " \
                 f"vali_loss {vali_loss:5.8f} | test_loss {test_loss:5.8f} |"

    results_info = f""

    for key, value in results.items():
        if isinstance(value, float):
            results_info = results_info + f" {key} {value:5.4f} |"

    info = basic_info + results_info + "\n"

    with open(folder_path + "/results.txt", 'a') as f:
        f.write(info)


def save_gradients(folder_path, model, epoch=0):
    """
    保存每一轮的梯度
    :param folder_path:
    :param model:
    :param epoch
    :return:
    """
    # 计算每层的梯度范数
    grad_dict = dict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad = param.grad
            else:
                grad = torch.zeros(size=(1, 1))
            grad_norm = torch.norm(grad)
            cprint.err('Layer name: {:50s}, Gradient Norm - {}'.format(name, grad_norm.item()))
            grad_dict[name] = grad_norm.item()

    with open(folder_path + '/gradients.txt', 'a') as f:
        f.write('Epoch {}\n'.format(epoch))
        for name, grad_norm in grad_dict.items():
            f.write('Layer name: {:50s}, Gradient Norm - {}\n'.format(name, grad_norm))
        f.write('\n')


def save_parameters(model, epoch=0, writer=None):
    """
    :param model:
    :param epoch:
    :param writer:
    :return:
    """
    if writer is not None:
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, global_step=epoch)



