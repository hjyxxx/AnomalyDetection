import os
import time

import numpy as np
import torch
from cprint import cprint
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from data_provider.data_factory import data_provider
from exp.exp_basic import ExpBasic
from utils.early_stopping import EarlyStopping
from utils.logging import save_args, save_model, save_results, save_gradients, save_parameters
from utils.result import compute_result


class ExpRec(ExpBasic):
    def __init__(self, args):
        super(ExpRec, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        for param in model.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

        return model

    def _get_data(self, flag):
        """
        :param flag: train, vali, test
        :return:
        """
        dataset, dataloader = data_provider(self.args, flag)

        return dataset, dataloader

    def _get_optimizer(self):
        """
        获得优化器
        :return:
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def _get_scheduler(self, optimizer):
        """
        获得调度器
        :param optimizer:
        :return:
        """
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return scheduler

    def _get_criterion(self):
        """
        获取损失函数
        :return:
        """
        criterion = nn.MSELoss()
        return criterion

    def vali(self, dataset, dataloader, criterion):
        """
        验证
        :param dataset:
        :param dataloader:
        :param criterion:
        :return:
        """

        total_loss = []
        total_preds = []
        total_trues = []

        self.model.eval()

        with torch.no_grad():
            for i, (batch_rec_x, batch_rec_y, batch_pre_x, batch_pre_y) in enumerate(dataloader):
                batch_rec_x = batch_rec_x.float().to(self.device)
                batch_rec_y = batch_rec_y.float().to(self.device)

                # (N, C, L, V)-->(N, C, L, V)
                outputs = self.model(batch_rec_x)

                preds = outputs.detach().cpu()
                trues = batch_rec_y.detach().cpu()

                loss = criterion(preds, trues)

                total_loss.append(loss)
                total_preds.append(preds)
                total_trues.append(trues)

        total_loss = np.average(total_loss)
        total_preds = np.concatenate(total_preds, axis=0)
        total_trues = np.concatenate(total_trues, axis=0)

        return total_loss, total_preds, total_trues

    def train(self):
        """
        训练
        :return:
        """

        # 获取数据集
        train_dataset, train_loader = self._get_data(flag='train')
        vali_dataset, vali_loader = self._get_data(flag='vali')
        test_dataset, test_loader = self._get_data(flag='test')

        train_len = len(train_dataset) if train_dataset is not None else 0
        vali_len = len(vali_dataset) if vali_dataset is not None else 0
        test_len = len(test_dataset) if test_dataset is not None else 0

        cprint.info('train: {}'.format(train_len))
        cprint.info('vali: {}'.format(vali_len))
        cprint.info(('test: {}'.format(test_len)))

        # 优化器, 调度器, 损失函数
        optimizer = self._get_optimizer()
        scheduler = self._get_scheduler(optimizer)
        criterion = self._get_criterion()

        # 记录日志
        folder_path = self.args.folder_path
        save_args(folder_path, self.args)               # 保存参数
        save_model(folder_path, str(self.model))        # 保存模型结构
        writer = SummaryWriter(log_dir=folder_path + '/tensorboard/')

        # 早停策略
        early_stopping = EarlyStopping()

        cprint.info("**********************start train**********************")
        iter = 0
        for epoch in range(1, self.args.train_epochs + 1):
            cprint.info("=" * 90)
            cprint.info(f"| Epoch {epoch} start train |")
            train_loss = []     # 保存训练损失
            self.model.train()
            ep_start_time = time.time()
            for batch_id, (batch_rec_x, batch_rec_y, batch_pre_x, batch_pre_y) in enumerate(train_loader):
                batch_rec_x = batch_rec_x.float().to(self.device)
                batch_rec_y = batch_rec_y.float().to(self.device)

                optimizer.zero_grad()

                bt_start_time = time.time()

                # (N, C, T, V)-->(N, C, T, V)
                outputs = self.model(batch_rec_x)

                loss = criterion(outputs, batch_rec_y)

                train_loss.append(loss.item())

                bt_elapsed = (time.time() - bt_start_time) * 1000

                cprint.info("| Train | Epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batches {:5.2f} | loss {:5.8f} |".format(
                    epoch, batch_id, len(train_loader), optimizer.param_groups[0]['lr'], bt_elapsed, loss.item()
                ))

                loss.backward()
                optimizer.step()

                writer.add_scalar('Loss/Train', loss.item(), iter)
                iter += 1

            scheduler.step()

            ep_elapsed = (time.time() - ep_start_time) * 1000
            cprint.info("| Epoch {} end train | ms/epoch {:5.2f} | loss {:5.8f} |".format(epoch, ep_elapsed, np.mean(train_loss)))
            cprint.info("=" * 90)

            vali_loss, test_loss = None, None
            # 验证
            if vali_len != 0:
                cprint.info("=" * 90)
                cprint.info(f"| Epoch {epoch} start vali |")
                vali_loss, vali_preds, vali_trues = self.vali(vali_dataset, vali_loader, criterion)
                writer.add_scalar('Loss/Vali', vali_loss, epoch)
                cprint.info("| Epoch {} end vali | loss {:5.8f} |".format(epoch, vali_loss))
                cprint.info("=" * 90)

            # 测试
            if test_len != 0:
                cprint.info("=" * 90)
                cprint.info("| Epoch {} start test |".format(epoch))
                test_loss, test_preds, test_trues = self.vali(test_dataset, test_loader, criterion)
                writer.add_scalar('Loss/Test', test_loss, epoch)
                cprint.info("| Epoch {} end test | loss {:5.8f} |".format(epoch, test_loss))
                cprint.info("=" * 90)

            test_preds = test_preds.transpose((0, 2, 3, 1))  # (N, T, V, C)
            test_trues = test_trues.transpose((0, 2, 3, 1))  # (N, T, V, C)

            # 计算结果
            results, sc, gt = compute_result(test_preds, test_trues, test_dataset.seg_metas, test_dataset.label_dict)

            # 保存结果
            save_results(folder_path, np.mean(train_loss), vali_loss, test_loss,
                         lr=optimizer.param_groups[0]['lr'], results=results, epoch=epoch)
            save_gradients(folder_path, self.model, epoch=epoch)
            save_parameters(self.model, epoch=epoch, writer=writer)

            if self.args.data in ['asd', 'asd2']:
                score = -results['video']['percentile@80']['AUC@ROC']
            else:
                score = -results['clip']['percentile@80']['AUC@ROC']

            early_stopping(score, self.model, folder_path)
            if early_stopping.early_stop:
                cprint.warn('Early Stopping')
                break

    def test(self):
        """
        测试
        :param folder_path:
        :return:
        """
        train_dataset, train_loader = self._get_data(flag='train')
        test_dataset, test_loader = self._get_data(flag='test')

        folder_path = self.args.folder_path
        self.model.load_state_dict(torch.load(os.path.join(folder_path, 'weights', 'model.pth')))
        folder_path = folder_path.replace('/', '_').replace('\\', '_')

        # 创建文件夹
        folder_path = './test_results/' + folder_path + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        train_total_preds = []
        train_total_trues = []

        test_total_preds = []
        test_total_trues = []

        with torch.no_grad():
            for i, (batch_rec_x, batch_rec_y, batch_pre_x, batch_pre_y) in enumerate(train_loader):
                batch_rec_x = batch_rec_x.float().to(self.device)
                batch_rec_y = batch_rec_y.float().to(self.device)
                outputs = self.model(batch_rec_x)

                preds = outputs.detach().cpu().numpy()
                trues = batch_rec_y.detach().cpu().numpy()

                train_total_preds.append(preds)
                train_total_trues.append(trues)

        with torch.no_grad():
            for i, (batch_rec_x, batch_rec_y, _, _) in enumerate(test_loader):
                batch_rec_x = batch_rec_x.float().to(self.device)
                batch_rec_y = batch_rec_y.float().to(self.device)
                outputs = self.model(batch_rec_x)

                preds = outputs.detach().cpu().numpy()
                trues = batch_rec_y.detach().cpu().numpy()

                test_total_preds.append(preds)
                test_total_trues.append(trues)

        train_total_preds = np.concatenate(train_total_preds, axis=0).transpose((0, 2, 3, 1))     # (N, T, V, C)
        train_total_trues = np.concatenate(train_total_trues, axis=0).transpose((0, 2, 3, 1))   # (N, T, V, C)

        test_total_preds = np.concatenate(test_total_preds, axis=0).transpose((0, 2, 3, 1))       # (N, T, V, C)
        test_total_trues = np.concatenate(test_total_trues, axis=0).transpose((0, 2, 3, 1))     # (N, T, V, C)

        # 保存结果
        np.save(folder_path + 'train_preds.npy', train_total_preds, allow_pickle=True)
        np.save(folder_path + 'train_trues.npy', train_total_trues, allow_pickle=True)
        np.save(folder_path + 'train_metas.npy', train_dataset.seg_metas, allow_pickle=True)

        np.save(folder_path + 'test_preds.npy', test_total_preds, allow_pickle=True)
        np.save(folder_path + 'test_trues.npy', test_total_trues, allow_pickle=True)
        np.save(folder_path + 'test_metas.npy', test_dataset.seg_metas, allow_pickle=True)

        # 计算score
        results, sc, gt = compute_result(test_total_preds, test_total_trues, test_dataset.seg_metas, test_dataset.label_dict)

        # print('| AUC@ROC {:5.4f} | AUC@PR {:5.4f} |'.format(
        #     results['AUC@ROC'], results['AUC@PR']
        # ))
        np.save(folder_path + 'test_grounds.npy', gt, allow_pickle=True)
        np.save(folder_path + 'test_scores.npy', sc, allow_pickle=True)
        np.save(folder_path + "test_label_dict", test_dataset.label_dict, allow_pickle=True)
        np.save(folder_path + "train_label_dict", train_dataset.label_dict, allow_pickle=True)
