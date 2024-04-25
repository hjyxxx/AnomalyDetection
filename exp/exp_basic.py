import os

import dill
import torch
from cprint import cprint

from models import PureGraph, TFWGraph, TFWModel, TModel
from models.bases import TCN, Transformer


class ExpBasic:
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TCN': TCN,
            'Transformer': Transformer,
            'PureGraph': PureGraph,
            'TFWGraph': TFWGraph,
            'TModel': TModel,
            'TFWModel': TFWModel

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

