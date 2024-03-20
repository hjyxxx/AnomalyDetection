import os

from torch.utils.data import DataLoader

from data_provider.data_utils import ae_trans_list
from datasets.dataset_pr import DatasetPR
from datasets.dataset_ws import DatasetWS


def data_provider(args, flag):
    """
    :param args:
    :param flag:
    :return:
    """
    # Base
    task_name = args.task_name      # 任务
    debug = args.debug              # Debug

    # DataLoader
    batch_size = args.batch_size        # 批次大小
    num_workers = args.num_workers
    pin_memory = args.pin_memory

    # Dataset
    path = os.path.join('data', args.data, args.data_path)      # 数据集路径
    seg_len = args.seg_len              # 步长
    seg_stride = args.seg_stride        # 步距
    for_len = args.for_len              # 预测长度

    transform = args.transform                  # 数据增强
    ckp = args.ckp

    # 归一化
    normalize_flag = args.normalize_flag
    vid_res = args.vid_res
    symm_range = args.symm_range
    sub_mean = args.sub_mean
    seg_conf_th = args.seg_conf_th

    divide = args.divide                # 数据集切分方式, random, file
    train_ratio = args.train_ratio      # 训练集比例
    vali_ratio = args.vali_ratio        # 验证集比例
    test_ratio = args.test_ratio        # 测试集比例
    train_txt_path = args.train_txt_path        # 训练集txt文件
    test_txt_path = args.test_txt_path          # 测试集txt文件
    data_shuffle = args.data_shuffle            # 是否打乱数据

    train_scenes = args.train_scenes
    vali_scenes = args.vali_scenes
    test_scenes = args.test_scenes


    if flag in ['test', 'vali']:
        shuffle = False
        transform_list = None
    else:
        shuffle = True
        if transform:
            transform_list = ae_trans_list
        else:
            transform_list = None

    if task_name == 'rec':
        dataset = DatasetPR(path=path,
                            seg_len=seg_len, seg_stride=seg_stride, for_len=0, transform_list=transform_list, ckp=ckp,
                            debug=debug, flag=flag,
                            normalize_flag=normalize_flag, vid_res=vid_res, symm_range=symm_range, sub_mean=sub_mean, seg_conf_th=seg_conf_th,
                            divide=divide, train_txt_path=train_txt_path, test_txt_path=test_txt_path,
                            train_scenes=train_scenes, vali_scenes=vali_scenes, test_scenes=test_scenes,
                            train_ratio=train_ratio, vali_ratio=vali_ratio, test_ratio=test_ratio, data_shuffle=data_shuffle,
                            task_name=task_name
                            )

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                pin_memory=pin_memory)
    elif task_name == 'pre':
        dataset = DatasetPR(path=path,
                            seg_len=seg_len, seg_stride=seg_stride, for_len=for_len, transform_list=transform_list, ckp=ckp,
                            debug=debug, flag=flag,
                            normalize_flag=normalize_flag, vid_res=vid_res, symm_range=symm_range, sub_mean=sub_mean, seg_conf_th=seg_conf_th,
                            divide=divide, train_txt_path=train_txt_path, test_txt_path=test_txt_path,
                            train_scenes=train_scenes, vali_scenes=vali_scenes, test_scenes=test_scenes,
                            train_ratio=train_ratio, vali_ratio=vali_ratio, test_ratio=test_ratio, data_shuffle=data_shuffle,
                            task_name=task_name
                            )

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                pin_memory=pin_memory)
    elif task_name == 'pr':
        dataset = DatasetPR(path=path,
                            seg_len=seg_len, seg_stride=seg_stride, for_len=for_len, transform_list=transform_list, ckp=ckp,
                            debug=debug, flag=flag,
                            normalize_flag=normalize_flag, vid_res=vid_res, symm_range=symm_range, sub_mean=sub_mean, seg_conf_th=seg_conf_th,
                            divide=divide, train_txt_path=train_txt_path, test_txt_path=test_txt_path,
                            train_scenes=train_scenes, vali_scenes=vali_scenes, test_scenes=test_scenes,
                            train_ratio=train_ratio, vali_ratio=vali_ratio, test_ratio=test_ratio, data_shuffle=data_shuffle,
                            task_name=task_name
                            )

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                pin_memory=pin_memory)
    elif task_name == 'ws':
        dataset = DatasetWS(path=path,
                            seg_len=seg_len, seg_stride=seg_stride, for_len=0, transform_list=transform_list, ckp=ckp,
                            debug=debug, flag=flag,
                            normalize_flag=normalize_flag, vid_res=vid_res, symm_range=symm_range, sub_mean=sub_mean, seg_conf_th=seg_conf_th,
                            divide=divide, train_txt_path=train_txt_path, test_txt_path=test_txt_path,
                            train_scenes=train_scenes, vali_scenes=vali_scenes, test_scenes=test_scenes,
                            train_ratio=train_ratio, vali_ratio=vali_ratio, test_ratio=test_ratio, data_shuffle=data_shuffle,
                            task_name=task_name
                            )

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                pin_memory=pin_memory)
    else:
        raise ValueError("Do Not Exist This Value: {}".format(task_name))

    return dataset, dataloader





