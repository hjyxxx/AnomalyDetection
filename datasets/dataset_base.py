from torch.utils.data import Dataset


class DatasetBase(Dataset):
    def __init__(self, path, seg_len, seg_stride, for_len, transform_list=None, ckp=None,
                 normalize_flag=0, vid_res=None, symm_range=True, sub_mean=False, seg_conf_th=0.0,
                 debug=False, flag='train',
                 divide='random', train_txt_path=None, vali_txt_path=None, test_txt_path=None,
                 train_scenes=None, vali_scenes=None, test_scenes=None,
                 train_ratio=1, vali_ratio=0, test_ratio=0, data_shuffle=False,
                 task_name='rec'):
        super(DatasetBase, self).__init__()

        self.path = path                        # 数据集路径
        self.seg_len = seg_len                  # 步长
        self.seg_stride = seg_stride            # 步距
        self.for_len = for_len                  # 预测长度

        self.debug = debug
        self.flag = flag                        # train or vali or test

        self.task_name = task_name

        # 数据增强
        self.transform_list = transform_list
        if transform_list is None or transform_list == []:
            self.num_transform = 1
            self.apply_transform = False
        else:
            self.num_transform = len(transform_list)
            self.apply_transform = True

        self.ckp = ckp

        # 数据集归一化参数
        self.vid_res = vid_res
        self.symm_range = symm_range
        self.sub_mean = sub_mean
        self.normalize_flag = normalize_flag
        self.seg_conf_th = seg_conf_th if flag == 'train' else 0.0

        # 数据集切分参数
        self.divide = divide                    # 数据集切割方式
        self.data_shuffle = data_shuffle        # 是否shuffle数据集
        self.train_ratio = train_ratio          # 重构/预测: 正常视频切分到训练集的比例
        self.vali_ratio = vali_ratio            # 重构/预测: 正常视频切分到验证集的比例
        self.test_ratio = test_ratio            # 重构/预测: 正常视频切分到测试集的比例

        self.train_scenes = train_scenes
        self.vali_scenes = vali_scenes
        self.test_scenes = test_scenes

        self.train_txt_path = train_txt_path
        self.vali_txt_path = vali_txt_path
        self.test_txt_path = test_txt_path

        # 读取正常的视频文件
        self.normal_videos, self.normal_scene_dict = self._read_video(flag='normal')
        # 读取异常的视频文件
        self.abnormal_videos, self.abnormal_scene_dict = self._read_video(flag='abnormal')

        # 切分数据集
        self.train_videos, self.vali_videos, self.test_videos = self._divide_dataset()

        # 得到指定场景的数据集
        self._select_scene()

        # 读取数据集
        self._read_data()

        # 读取标签
        self._read_label()

        self.num_samples = len(self.seg_data_np)

    def _read_video(self, flag):
        """
        读取视频列表
        :param flag: normal or abnormal
        :return:
        """
        raise NotImplementedError

    def _divide_dataset(self):
        """
        划分数据集
        :return:
        """
        raise NotImplementedError

    def _read_data(self):
        """
        读取数据
        :return:
        """
        raise NotImplementedError

    def _read_label(self):
        """
        读取标签
        :return:
        """
        raise NotImplementedError

    def _select_scene(self):
        """
        :return:
        """
        if self.train_scenes is not None and self.train_scenes:
            self.train_videos = [video for video in self.train_videos if
                                 int(video.split('/')[1].split('_')[0]) in self.train_scenes]

        if self.vali_scenes is not None and self.vali_scenes:
            self.vali_videos = [video for video in self.vali_videos if
                                 int(video.split('/')[1].split('_')[0]) in self.vali_scenes]

        if self.test_scenes is not None and self.test_scenes:
            self.test_videos = [video for video in self.test_videos if
                                 int(video.split('/')[1].split('_')[0]) in self.test_scenes]

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
