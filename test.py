import os
import random

import numpy as np
import pywt
import torch.fft
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import norm
import seaborn as sns

from utils.cal_metrics import cal_roc_auc, cal_pr_auc
from utils.result import get_scores, smooth, min_max, compute_result

random.seed(2024)


def test_video(preds, trues, metas, label_dict, video_name):
    """
    测试一个视频
    :param preds: (N, T, V, C)
    :param trues: (N, T, V, C)
    :param metas: (N, 4)
    :param label_dict:
    :param video_name: sceneId_videoId
    :return:
    """
    assert video_name in label_dict.keys()

    scene_id, video_id = video_name.split('_')
    video_label = np.array(label_dict[video_name])

    video_metas_index = np.where((metas[:, 0] == scene_id) &
                                 (metas[:, 1] == video_id))[0]
    video_metas = metas[video_metas_index]
    video_person_ids = set([meta[2] for meta in video_metas])

    score_zeros = np.zeros(video_label.shape[0])
    video_person_scores_dict = {i: np.copy(score_zeros) for i in video_person_ids}

    # 对每一个人进行遍历
    for person_id in video_person_ids:
        person_meta_index = np.where((metas[:, 0] == scene_id) &
                                     (metas[:, 1] == video_id) &
                                     (metas[:, 2] == person_id))[0]

        person_pred = preds[person_meta_index]      # (N, L, V, C)
        person_true = trues[person_meta_index]      # (N, L, V, C)

        person_scores = get_scores(person_pred, person_true)    # (N, )

        person_frames_sample = np.array([metas[i][3] for i in person_meta_index])
        person_frames = person_frames_sample[:, -1]     # (N, )
        video_person_scores_dict[person_id][person_frames] = person_scores

    video_person_scores = np.stack(list(video_person_scores_dict.values()))
    video_scores = np.amax(video_person_scores, axis=0)

    # # 平滑分数
    # video_scores = smooth(video_scores, 10)
    #
    # # 最大最小归一化
    # video_scores = min_max(video_scores)

    # mean = video_scores.mean()
    # std = video_scores.std()
    # video_scores = (video_scores - mean) / std
    # #
    video_scores = smooth(video_scores, 30)

    # 绘图
    # _, ax = plt.subplots(1, 1)
    # ax.plot(video_scores, color='red')
    # # ax.plot(video_label, color='green')
    # # ax.set_ylim(0, 1)
    # plt.show()

    return video_scores


def compute_scores(preds, trues, metas, label_dict):
    """
    计算结果
    :param preds: ndarray, (N, T, V, C)
    :param trues: ndarray, (N, T, V, C)
    :param metas: list, [[sceneId, videoId, personId, [frameIds]]], length=N
    :param label_dict: {sceneId_videoId: []}
    :return:
    """
    metas = np.array(metas, dtype=object)
    scenes = set(meta[0] for meta in metas)        # 获得所有场景Id
    scores = {}
    scene_dict = {scene: set(meta[1] for meta in metas if meta[0] == scene) for scene in scenes}    # 获得每个场景下的视频Id
    clip_labels = []     # 保存片段标签
    clip_scores = []     # 保存片段分数
    video_labels = []       # 保存视频标签
    video_scores = []       # 保存视频分数

    # 遍历每个场景
    for scene_id in scene_dict.keys():
        scene_clip_scores = []       # 保存场景的片段分数
        scene_clip_labels = []       # 保存场景的片段标签
        scene_video_scores = []      # 保存场景的视频分数
        scene_video_labels = []      # 保存场景的视频标签

        # 遍历每个视频
        for video_id in scene_dict[scene_id]:
            # 得到该视频每一个片段的标签
            video_info = f"{scene_id}_{video_id}"
            video_clip_labels = label_dict[video_info]
            video_clip_labels = np.array(video_clip_labels)

            # 得到视频标签
            video_label = np.ones(1) if 1 in video_clip_labels else np.zeros(1)

            # 得到该视频的meta信息
            video_metas_index = np.where((metas[:, 0] == scene_id) &
                                         (metas[:, 1] == video_id))[0]
            video_metas = metas[video_metas_index]

            # 得到该视频中出现的人id
            video_person_ids = set([meta[2] for meta in video_metas])

            score_zeros = np.zeros(video_clip_labels.shape[0])
            video_person_scores_dict = {i: np.copy(score_zeros) for i in video_person_ids}

            # 对每一个人进行遍历
            for person_id in video_person_ids:
                person_metas_index = np.where((metas[:, 0] == scene_id) &
                                              (metas[:, 1] == video_id) &
                                              (metas[:, 2] == person_id))[0]

                person_preds = preds[person_metas_index]        # (N, T, V, C)
                person_trues = trues[person_metas_index]        # (N, T, V, C)

                person_scores = get_scores(person_trues, person_preds)      # (N, )

                person_frames_sample = np.array([metas[i][3] for i in person_metas_index])      # (N, L)
                person_frames = person_frames_sample[:, -1]     # (N, )

                video_person_scores_dict[person_id][person_frames] = person_scores

            video_person_scores = np.stack(list(video_person_scores_dict.values()))
            video_clip_scores = np.amax(video_person_scores, axis=0)

            # 平滑分数
            # video_clip_scores = smooth(video_clip_scores, 40)
            # 最大最小归一化
            # video_clip_scores = min_max(video_clip_scores)

            # video_mean = video_clip_scores.mean()
            # video_std = video_clip_scores.std()
            # video_clip_scores = (video_clip_scores - video_mean) / video_std

            video_mean = np.mean(video_clip_scores)

            video_clip_scores = smooth(video_clip_scores, 30)

            scene_clip_scores.append(video_clip_scores)
            scene_clip_labels.append(video_clip_labels)

            scene_video_scores.append(video_mean)
            scene_video_labels.append(video_label)

        scene_clip_scores = np.concatenate(scene_clip_scores)
        scene_clip_labels = np.concatenate(scene_clip_labels)
        scene_video_labels = np.concatenate(scene_video_labels)
        scene_video_scores = np.array(scene_video_scores)

        # scene_clip_scores = min_max(scene_clip_scores)

        mean = scene_clip_scores.mean()
        std = scene_clip_scores.std()
        scene_clip_scores = (scene_clip_scores - mean) / std

        clip_scores.append(scene_clip_scores)
        clip_labels.append(scene_clip_labels)
        video_scores.append(scene_video_scores)
        video_labels.append(scene_video_labels)

    clip_scores = np.concatenate(clip_scores)
    clip_labels = np.concatenate(clip_labels)
    video_scores = np.concatenate(video_scores)
    video_labels = np.concatenate(video_labels)

    return clip_scores


def analyze(train_preds, train_trues, train_metas, train_label_dict, test_preds, test_trues, test_metas, test_label_dict, train_num, test_num):
    train_videos = list(train_label_dict.keys())
    test_videos = list(test_label_dict.keys())


    train_scores = compute_scores(train_preds, train_trues, train_metas, train_label_dict)

    test_scores = compute_scores(test_preds ,test_trues, test_metas, test_label_dict)


    train_videos = random.sample(train_videos, train_num)

    test_videos = random.sample(test_videos, test_num)

    res = []

    for video in train_videos:
        scores = test_video(train_preds, train_trues, train_metas, train_label_dict, video)

        res.append(scores)

    for video in test_videos:
        scores = test_video(test_preds, test_trues, test_metas, test_label_dict, video)

        res.append(scores)

    res_np = np.concatenate(res)

    v = 0
    vlines = []
    for i in res:
        v = v + len(i)
        vlines.append(v)

    _, ax = plt.subplots(1, 1)
    ax.plot(res_np, color='blue')
    ax.vlines(vlines, 0, 0.0005, linestyles='dashed', colors='red')
    plt.show()





if __name__ == '__main__':

    folder_path = 'checkpoints/rec/TFWModel/shtc/train/save0'
    results_folder_path = os.path.join('test_results', folder_path.replace('/', '_'))

    train_trues = np.load(os.path.join(results_folder_path, 'train_trues.npy'), allow_pickle=True)
    train_preds = np.load(os.path.join(results_folder_path, 'train_preds.npy'), allow_pickle=True)
    train_metas = np.load(os.path.join(results_folder_path, 'train_metas.npy'), allow_pickle=True)

    test_trues = np.load(os.path.join(results_folder_path, 'test_trues.npy'), allow_pickle=True)
    test_preds = np.load(os.path.join(results_folder_path, 'test_preds.npy'), allow_pickle=True)
    test_metas = np.load(os.path.join(results_folder_path, 'test_metas.npy'), allow_pickle=True)

    test_grounds = np.load(os.path.join(results_folder_path, 'test_grounds.npy'), allow_pickle=True)
    test_scores = np.load(os.path.join(results_folder_path, 'test_scores.npy'), allow_pickle=True)
    test_label_dict = np.load(os.path.join(results_folder_path, 'test_label_dict.npy'), allow_pickle=True).item()
    train_label_dict = np.load(os.path.join(results_folder_path, 'train_label_dict.npy'), allow_pickle=True).item()

    train_scores = get_scores(train_trues, train_preds)

    # u = train_scores.mean()
    # std = train_scores.std()

    results, _, _ = compute_result(test_preds, test_trues, test_metas, test_label_dict)
    print(results)

    # test_video(train_preds, train_trues, train_metas, train_label_dict, video_name='01_015')
    # test_video(test_preds, test_trues, test_metas, test_label_dict, video_name='01_0014')

    # analyze(train_preds, train_trues, train_metas, train_label_dict,
    #         test_preds, test_trues, test_metas, test_label_dict,
    #         train_num=10, test_num=10)

    data = train_trues[:, -1, :, :]
    sns.displot(data[:, 1, 1])
    (mu, sigma) = norm.fit(data[:, 1, 1])
    print('mu={:.2f} and sigma={:.2f}'.format(mu, sigma))

    fig = plt.figure()
    res = stats.probplot(data[:, 1, 1], plot=plt)
    plt.show()

