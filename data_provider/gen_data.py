import json
import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns


def gen_data(path, video_list, seg_len, seg_stride, vid_res, symm_range, sub_mean, normalize_flag, seg_conf_th):
    """
    :param path: {data}/{datatype}/
    :param video_list: [normal/01_001_alphapose_tracked_person.json]
    :param seg_len: 步长
    :param seg_stride: 步距
    :param vid_res: 分辨率, [1920, 1080]
    :param symm_range:
    :param sub_mean:
    :param normalize_flag:
    :param seg_conf_th:
    :return:
    """
    seg_data_np = []            # 保存滑窗后的数据
    seg_metas = []              # 保存滑窗后的信息, [[scene_id, video_id, person_id, [frames]]]

    for video in tqdm(video_list):
        flag, name = video.split('/')

        scene_id, video_id = name.split('_')[:2]        # scene_id: 场景id, video_id: 视频id

        video_path = os.path.join(path, video)

        with open(video_path, 'r') as f:
            video_dict = json.load(f)

        video_seg_data_np, video_seg_metas = gen_video_data(video_dict, seg_len, seg_stride, scene_id, video_id)

        seg_data_np.append(video_seg_data_np)
        seg_metas += video_seg_metas

    seg_data_np = np.concatenate(seg_data_np, axis=0)       # (N, L, V, C)

    # 归一化
    seg_data_np = normalize(seg_data_np, vid_res, symm_range=symm_range, sub_mean=sub_mean, flag=normalize_flag)

    # 将opencv检测的17个姿态点转换至18个姿态点
    if seg_data_np.shape[-2] == 17:
        seg_data_np = keypoints17_to_coco18(seg_data_np)

    # N, L, V, C --> N, C, L, V
    seg_data_np = np.transpose(seg_data_np, (0, 3, 1, 2)).astype(np.float32)

    if seg_conf_th > 0.0 and seg_data_np.shape[1] == 3:
        seg_data_np, seg_metas = seg_conf_th_filter(seg_data_np, seg_metas, seg_conf_th)

    # 去除置信度
    if seg_data_np.shape[1] == 3:
        seg_data_np = seg_data_np[:, :2, ...]

    return seg_data_np, seg_metas


def gen_video_data(person_dict, seg_len, seg_stride, scene_id, video_id):
    """
    生成一个视频中所有人滑窗后姿态点数据
    :param person_dict: dict, {'person_id': {'frame_id': {'keypoints': [[x,y,c], []]}, 'confidence': float, 'box': []}}}
    :param seg_len: int, 窗长
    :param seg_stride: int, 步距
    :param scene_id: str, 场景id
    :param video_id: str, 视频id
    :return:
    """

    video_seg_data_np = []      # 保存一个视频中所有人的滑窗数据
    video_seg_metas = []        # 保存一个视频中所有人滑窗后的信息

    person_ids = person_dict.keys()
    for person_id in sorted(person_ids, key=lambda x: int(x)):
        single_person = person_dict[person_id]      # 取出这个人在这个视频中的姿态点轨迹
        single_person_data_np = []                  # 保存这个人在视频中的轨迹信息
        single_person_frames = sorted(single_person.keys(), key=lambda x: int(x))

        for frame_id in single_person_frames:
            content = single_person[frame_id]

            keypoints = np.array(content['keypoints'])      # (V, C)
            confidence = content['confidence']
            box = content['box']

            single_person_data_np.append(keypoints)

        single_person_data_np = np.stack(single_person_data_np, axis=0)

        # 对每个人的姿态点进行滑窗
        person_seg_data_np, person_seg_metas = split_pose_to_seg(
            single_person_data_np, single_person_frames, seg_len, seg_stride, scene_id, video_id, person_id)

        video_seg_data_np.append(person_seg_data_np)
        video_seg_metas += person_seg_metas

    video_seg_data_np = np.concatenate(video_seg_data_np, axis=0)       # (N, L, V, C)

    return video_seg_data_np, video_seg_metas


def split_pose_to_seg(single_person_data_np, single_person_frames, seg_len, seg_stride, scene_id, video_id, person_id):
    """
    对一个人的姿态点数据进行滑窗
    :param single_person_data_np: (N, V, C)
    :param single_person_frames: list, [frames]
    :param seg_len:
    :param seg_stride:
    :param scene_id:
    :param video_id:
    :param person_id:
    :return:
    """

    video_t, kp_count, kp_dim = single_person_data_np.shape     # 轨迹长度, 姿态点个数, 特征维度

    pose_seg_data_np = np.empty((0, seg_len, kp_count, kp_dim))     # 保存滑窗后的数据
    pose_seg_metas = []         # 保存滑窗后的信息, [scene_id, video_id, person_id, [frames]], frames是这个窗口的帧序号

    num_seg = (np.floor((video_t - seg_len) / seg_stride) + 1).astype(np.int_)      # 窗口个数

    single_person_frames = sorted(int(i) for i in single_person_frames)     # 将帧序号转换为int类型

    for i in range(num_seg):
        start = i * seg_stride      # 每个窗口的起始位置, 0, seg_stride, 2*seg_stride
        start_frame = single_person_frames[start]       # 起始位置对应的帧序号

        if is_seg(single_person_frames, start_frame, seg_len):
            curr_seg_data_np = single_person_data_np[start: start + seg_len].reshape(1, seg_len, kp_count, kp_dim)
            curr_seg_frames = single_person_frames[start: start + seg_len]
            pose_seg_data_np = np.append(pose_seg_data_np, curr_seg_data_np, axis=0)
            pose_seg_metas.append([scene_id, video_id, int(person_id), curr_seg_frames])

    return pose_seg_data_np, pose_seg_metas


def is_seg(single_person_frames, start_frame, seg_len, missing_frames=2):
    """
    判断是否满足一个窗口
    :param single_person_frames: list, int
    :param start_frame: int, 开始帧序号
    :param seg_len: int, 窗长
    :param missing_frames: int, 最大缺失帧数
    :return: Ture or False
    """

    start_frame_index = single_person_frames.index(start_frame)     # 起始帧在帧序列中的位置
    excepted_ids = list(range(start_frame, start_frame + seg_len))      # 期待的一个窗口的帧序列号
    act_ids = single_person_frames[start_frame_index: start_frame_index + seg_len]      # 真实的一个窗口的帧序列号
    min_overlap = seg_len - missing_frames      # 最少要求的连续帧个数
    key_overlap = len(set(act_ids).intersection(excepted_ids))
    if key_overlap > min_overlap:
        return True
    else:
        return False


def normalize(seg_data_np, vid_res, symm_range=True, sub_mean=False, flag=1):
    """
    预处理
    :param seg_data_np: (N, L, V, C)
    :param vid_res: list, [856, 480]
    :param symm_range:
    :param sub_mean:
    :param flag:
    :return:
    """
    if flag == 0:
        if seg_data_np.shape[-1] == 3:
             vid_res = vid_res + [1]
        norm_factor = np.array(vid_res)
        seg_data_np_normalized = seg_data_np / norm_factor
        if symm_range:
            seg_data_np_normalized[..., :2] = 2 * seg_data_np_normalized[..., :2] - 1

        if sub_mean:
            mean_kp_val = np.mean(seg_data_np_normalized[..., :2], (1, 2))
            seg_data_np_normalized[..., :2] -= mean_kp_val[:, None, None, :]
    else:
        seg_data_np_normalized = seg_data_np

    return seg_data_np_normalized


def keypoints17_to_coco18(kps):
    """
    将 17 个关键点 coco 格式的框架转换为 18 个关键点的框架。
    新关键点（颈部）是肩膀的平均值，点也被重新排序
    :param kps: (N, L, V, C)
    :return: (N, L, V, C)
    """
    kp_np = np.array(kps)
    neck_kp_vec = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
    kp_np = np.concatenate([kp_np, neck_kp_vec[..., None, :]], axis=-2)
    opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    opp_order = np.array(opp_order, dtype=np.int_)
    kp_coco18 = kp_np[..., opp_order, :]
    return kp_coco18


def seg_conf_th_filter(seg_data_np, seg_meta, seg_conf_th):
    """
    过滤置信度低的窗口段
    :param seg_data_np: ndarray, (N, C, L, V)
    :param seg_meta: list, [scene_id, video_id, person_id, start_frame]
    :param seg_conf_th: float
    :return:
    """
    seg_len = seg_data_np.shape[2]
    conf_vals = seg_data_np[:, 2]       # (N, L, V)
    sum_confs = conf_vals.sum(axis=(1, 2)) / seg_len        # (N)

    # sns.displot(sum_confs)
    # sns.displot(sum_confs, kind='kde')
    # sns.displot(sum_confs, kind='ecdf')

    # plt.show()

    seg_data_filter = seg_data_np[sum_confs > seg_conf_th]
    seg_meta_filter = np.array(seg_meta)[sum_confs > seg_conf_th].tolist()
    return seg_data_filter, seg_meta_filter
