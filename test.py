import os

import numpy as np
import pywt
import torch.fft
from matplotlib import pyplot as plt

from utils.result import get_scores, smooth, min_max, compute_result


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

    # 平滑分数
    video_scores = smooth(video_scores, 10)

    # 最大最小归一化
    video_scores = min_max(video_scores)

    # 绘图
    _, ax = plt.subplots(1, 1)
    ax.plot(video_scores, color='red')
    ax.plot(video_label, color='green')
    # ax.set_ylim(0, 1)
    plt.show()


if __name__ == '__main__':

    folder_path = 'checkpoints/rec/PureGraph/asd/train/save1'
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

    results, _, _ = compute_result(test_preds, test_trues, test_metas, test_label_dict)
    print(results)

    # test_video(train_preds, train_trues, train_metas, train_label_dict, video_name='01_015')
    # test_video(test_preds, test_trues, test_metas, test_label_dict, video_name='01_0014')

