import numpy as np
from scipy.ndimage import gaussian_filter1d

from utils.cal_metrics import cal_roc_auc, cal_pr_auc, cal_f1_acc


def compute_result(preds, trues, metas, label_dict):
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

    scene_dict = {scene: set(meta[1] for meta in metas if meta[0] == scene) for scene in scenes}    # 获得每个场景下的视频Id

    labels = []     # 保存标签
    scores = []     # 保存分数

    # 遍历每个场景
    for scene_id in scene_dict.keys():
        scene_scores = []       # 保存场景的分数
        scene_labels = []       # 保存场景的标签

        # 遍历每个视频
        for video_id in scene_dict[scene_id]:
            # 得到该视频每一帧的标签
            video_info = f"{scene_id}_{video_id}"
            video_label = label_dict[video_info]
            video_label = np.array(video_label)

            # 得到该视频的meta信息
            video_metas_index = np.where((metas[:, 0] == scene_id) &
                                        (metas[:, 1] == video_id))[0]
            video_metas = metas[video_metas_index]

            # 得到该视频中出现的人id
            video_person_ids = set([meta[2] for meta in video_metas])

            score_zeros = np.zeros(video_label.shape[0])
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
            video_scores = np.amax(video_person_scores, axis=0)

            # 平滑分数
            video_scores = smooth(video_scores, 40)
            # 最大最小归一化
            # video_scores = min_max(video_scores)

            mean = video_scores.mean()
            std = video_scores.std()
            video_scores = (video_scores - mean) / std
            #
            # video_scores = smooth(video_scores, 20)

            scene_scores.append(video_scores)
            scene_labels.append(video_label)

        scene_scores = np.concatenate(scene_scores)
        scene_labels = np.concatenate(scene_labels)

        # scene_scores = min_max(scene_scores)

        # mean = scene_scores.mean()
        # std = scene_scores.std()
        # scene_scores = (scene_scores - mean) / std

        scores.append(scene_scores)
        labels.append(scene_labels)

    scores = np.concatenate(scores)
    labels = np.concatenate(labels)

    mean = scores.mean()
    std = scores.std()
    scores = (scores - mean) / std
    # scores = min_max(scores)

    fpr, tpr, roc_auc = cal_roc_auc(scores, labels)
    _, _, pr_auc = cal_pr_auc(scores, labels)
    acc, f1, precision, recall = cal_f1_acc(scores, labels)

    results = {
        "fpr": fpr,
        "tpr": tpr,
        "AUC@ROC": roc_auc,
        "AUC@PR": pr_auc,
        "F1": f1,
        "ACC": acc,
        "precision": precision,
        "recall": recall
    }

    return results, scores, labels


def get_scores(trues, preds):
    """
    计算异常分数
    :param trues: ndarray, (N, L, V, C)
    :param preds: ndarray, (N, L, V, C)
    :return:
    """

    mse = np.mean((trues - preds) ** 2, axis=(-3, -2, -1))      # (N, )
    scores = mse

    return scores


def smooth(y, n):
    """
    :param y: (N, )
    :param n:
    :return:
    """

    y_smooth = gaussian_filter1d(y, sigma=n)

    # win = np.ones(n) / n
    # y = np.array([y[0]] * (n // 2) + y.tolist() + [y[-1]] * (n // 2))
    #
    # y_smooth = np.convolve(y, win, mode='valid')

    return y_smooth


def min_max(y):
    """
    :param y:
    :return:
    """

    if y.max() == y.min():
        return y

    return (y - y.min()) / (y.max() - y.min())






