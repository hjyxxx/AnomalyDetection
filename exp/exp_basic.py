import os

import torch
from cprint import cprint

from models import PureGraph
from models.bases import TCN, Transformer


class ExpBasic:
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TCN': TCN,
            'Transformer': Transformer,
            'PureGraph': PureGraph

        }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self._info()

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        """
        设置GPU或者CPU运行
        :return:
        """

        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            cprint.info("Use GPU: {}".format(self.args.gpu))
        else:
            device = torch.device('cpu')
            cprint.info('Use CPU')

        return device

    def _info(self):
        """
        打印模型信息
        :return:
        """
        # 计算模型参数个数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        cprint.info("Total Parameters: {}".format(total_params))
        cprint.info("Trainable Parameters: {}".format(trainable_params))

    def _create_folder(self, flag='train'):
        """
        创建保存的文件夹
        :param flag:
        :return:
        """
        path = os.path.join(self.args.checkpoints, self.args.task_name, self.args.model, self.args.data, flag)
        if not os.path.exists(path):
            save_path = os.path.join(path, 'save0')
            os.makedirs(save_path)
        else:
            save_folders = os.listdir(path)
            if save_folders:
                sorted_folders = sorted(save_folders, key=lambda x: int(x[4:]))
                last_folders = sorted_folders[-1]
                last_num = int(last_folders[4:])
                name = "save" + str(last_num + 1)
            else:
                name = 'save0'
            save_path = os.path.join(path, name)
            os.makedirs(save_path)
        cprint.info('logging: {}'.format(save_path))

        return save_path
