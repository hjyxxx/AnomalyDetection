import math
import os
import random

import numpy as np
import torch

from data_provider.gen_data import gen_data
from datasets.dataset_base import DatasetBase


class DatasetPR(DatasetBase):
    def __init__(self, path, seg_len, seg_stride, for_len, transform_list=None, ckp=None,
                 normalize_flag=0, vid_res=None, symm_range=True, sub_mean=False, seg_conf_th=0.0,
                 debug=False, flag='train',
                 divide='random', train_txt_path=None, vali_txt_path=None, test_txt_path=None,
                 train_scenes=None, vali_scenes=None, test_scenes=None,
                 train_ratio=1, vali_ratio=0, test_ratio=0, data_shuffle=False,
                 task_name='rec'):
        super(DatasetPR, self).__init__(path, seg_len, seg_stride, for_len, transform_list, ckp,
                                        normalize_flag, vid_res, symm_range, sub_mean, seg_conf_th,
                                        debug, flag,
                                        divide, train_txt_path, vali_txt_path, test_txt_path,
                                        train_scenes, vali_scenes, test_scenes,
                                        train_ratio, vali_ratio, test_ratio, data_shuffle,
                                        task_name)

    def _read_video(self, flag):
        """
        :param flag: 'normal' or 'abnormal'
        :return: videos: [01_001_openpose_tracked_person.json, ...]
                 scene_dict: {01: [001, 002, ...], ...}
        """

        path = os.path.join(self.path, flag)
        videos = os.listdir(path)

        scenes = set([video.split('_')[0] for video in videos])     # 场景
        scenes = sorted(scenes)

        scene_dict = {scene: [video for video in videos if video.startswith(scene)] for scene in scenes}

        return videos, scene_dict

    def _divide_dataset(self):
        """
        :return:
        """
        if self.divide == 'random':
            train_videos, vali_videos, test_videos = self._divide_dataset_by_random()
        else:
            train_videos, vali_videos, test_videos = self._divide_dataset_by_random()

        if self.debug:
            train_videos = train_videos[:5]
            vali_videos = vali_videos[:5]
            test_videos = test_videos[:5]

        return train_videos, vali_videos, test_videos

    def _divide_dataset_by_random(self):
        """
        随机切割数据集
        :return:
        """
        train_videos = []
        vali_videos = []
        test_videos = []

        normal_scenes = sorted(self.normal_scene_dict.keys())
        abnormal_scenes = sorted(self.abnormal_scene_dict.keys())

        for scene in normal_scenes:
            videos = self.normal_scene_dict[scene]
            if self.data_shuffle:
                random.shuffle(videos)

            train_start = 0
            train_end = math.ceil(len(videos) * self.train_ratio)

            vali_start = train_end
            vali_end = vali_start + math.ceil(len(videos) * self.vali_ratio)

            test_start = vali_end
            test_end = len(videos)

            train_videos.extend(['normal/' + video for video in videos[train_start: train_end]])
            vali_videos.extend(['normal/' + video for video in videos[vali_start: vali_end]])
            test_videos.extend(['normal/' + video for video in videos[test_start: test_end]])

        for scene in abnormal_scenes:
            videos = self.abnormal_scene_dict[scene]
            if self.data_shuffle:
                random.shuffle(videos)
            test_videos.extend(['abnormal/' + video for video in videos])

        if self.data_shuffle:
            random.shuffle(train_videos)
            random.shuffle(vali_videos)
            random.shuffle(test_videos)

        return train_videos, vali_videos, test_videos

    def _read_data(self):
        """
        :return:
        """
        if self.flag == 'train':
            video_list = self.train_videos
        elif self.flag == 'vali':
            video_list = self.vali_videos
        elif self.flag == 'test':
            video_list = self.test_videos
        else:
            raise ValueError("Do Not Exist This Value: {}".format(self.flag))

        if not video_list:
            self.seg_data_np = []
            self.seg_metas = []
            return

        # if self.task_name == 'pre':
        #     seg_len = self.seg_len + self.for_len
        # else:
        #     seg_len = self.seg_len
        self.seg_data_np, self.seg_metas = gen_data(self.path, video_list, self.seg_len+self.for_len, self.seg_stride,
                                                    self.vid_res, self.symm_range, self.sub_mean, self.normalize_flag, self.seg_conf_th)

        if self.ckp is not None and self.ckp:
            self.seg_data_np = self.seg_data_np[..., self.ckp]

    def _read_label(self):
        """
        读取标签
        :return:
        """
        if self.flag == 'train':
            video_list = self.train_videos
        elif self.flag == 'vali':
            video_list = self.vali_videos
        elif self.flag == 'test':
            video_list = self.test_videos
        else:
            raise ValueError("Do Not Exist This Value: {}".format(self.flag))

        self.label_dict = dict()

        for video in video_list:
            video_class = video.split('/')[0]
            scene_id, video_id = video.split('/')[1].split('_')[:2]
            video_info = f"{scene_id}_{video_id}"
            label_path = self.path + f"/{video_class}_label/{video_info}.npy"
            label = np.load(label_path)

            self.label_dict[video_info] = label.tolist()

    def __len__(self):
        return self.num_samples * self.num_transform

    def __getitem__(self, index):
        if self.apply_transform:
            sample_index = index % self.num_samples
            trans_index = index // self.num_samples
            data_transformed = self.transform_list[trans_index](self.seg_data_np[sample_index])
        else:
            sample_index = index
            data_transformed = self.seg_data_np[sample_index]

        batch_rec_x = data_transformed[:, :self.seg_len]
        batch_rec_y = data_transformed[:, :self.seg_len]

        batch_pre_x = data_transformed[:, :self.seg_len]
        batch_pre_y = data_transformed[:, -self.for_len:]

        return batch_rec_x, batch_rec_y, batch_pre_x, batch_pre_y

