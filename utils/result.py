import numpy as np
from scipy.ndimage import gaussian_filter1d

from utils.cal_metrics import cal_roc_auc, cal_pr_auc, cal_f1_acc_by_percentile, cal_f1_acc_by_best


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

        scene_clip_scores = min_max(scene_clip_scores)

        # mean = scene_clip_scores.mean()
        # std = scene_clip_scores.std()
        # scene_clip_scores = (scene_clip_scores - mean) / std

        clip_scores.append(scene_clip_scores)
        clip_labels.append(scene_clip_labels)
        video_scores.append(scene_video_scores)
        video_labels.append(scene_video_labels)

    clip_scores = np.concatenate(clip_scores)
    clip_labels = np.concatenate(clip_labels)
    video_scores = np.concatenate(video_scores)
    video_labels = np.concatenate(video_labels)

    # mean = scores.mean()
    # std = scores.std()
    # scores = (scores - mean) / std
    # scores = min_max(scores)

    clip_fpr, clip_tpr, clip_roc_auc = cal_roc_auc(clip_scores, clip_labels)
    _, _, clip_pr_auc = cal_pr_auc(clip_scores, clip_labels)
    clip_threshold_p, clip_f1_p, clip_acc_p, clip_precision_p, clip_recall_p, clip_TP_p, clip_FP_p, clip_TN_p, clip_FN_p, clip_TPR_p, clip_FPR_p = cal_f1_acc_by_percentile(clip_scores, clip_labels)
    # clip_threshold_b, clip_f1_b, clip_acc_b, clip_precision_b, clip_recall_b, clip_TP_b, clip_FP_b, clip_TN_b, clip_FN_b, clip_TPR_b, clip_FPR_b = cal_f1_acc_by_best(clip_scores, clip_labels)

    video_fpr, video_tpr, video_roc_auc = cal_roc_auc(video_scores, video_labels)
    _, _, video_pr_auc = cal_pr_auc(video_scores, video_labels)
    video_threshold_p, video_f1_p, video_acc_p, video_precision_p, video_recall_p, video_TP_p, video_FP_p, video_TN_p, video_FN_p, video_TPR_p, video_FPR_p = cal_f1_acc_by_percentile(video_scores, video_labels)
    # video_threshold_b, video_f1_b, video_acc_b, video_precision_b, video_recall_b, video_TP_b, video_FP_b, video_TN_b, video_FN_b, video_TPR_b, video_FPR_b = cal_f1_acc_by_best(video_scores, video_labels)

    results = {
        "clip": {
            "percentile@80": {
                "AUC@ROC": clip_roc_auc.astype(float),
                "AUC@PR": clip_pr_auc.astype(float),
                "Threshold": clip_threshold_p.astype(float),
                "F1": clip_f1_p.astype(float),
                "ACC": clip_acc_p.astype(float),
                "precision": clip_precision_p.astype(float),
                "recall": clip_recall_p.astype(float),
                "TP": clip_TP_p.astype(float),
                "FP": clip_FP_p.astype(float),
                "TN": clip_TN_p.astype(float),
                "FN": clip_FN_p.astype(float),
                "TPR": clip_TPR_p.astype(float),
                "FPR": clip_FPR_p.astype(float)
            },
            # "best": {
            #     "AUC@ROC": clip_roc_auc.astype(float),
            #     "AUC@PR": clip_pr_auc.astype(float),
            #     "Threshold": clip_threshold_b.astype(float),
            #     "F1": clip_f1_b.astype(float),
            #     "ACC": clip_acc_b.astype(float),
            #     "precision": clip_precision_b.astype(float),
            #     "recall": clip_recall_b.astype(float),
            #     "TP": clip_TP_b.astype(float),
            #     "FP": clip_FP_b.astype(float),
            #     "TN": clip_TN_b.astype(float),
            #     "FN": clip_FN_b.astype(float),
            #     "TPR": clip_TPR_b.astype(float),
            #     "FPR": clip_FPR_b.astype(float)
            # }
        },
        "video": {
            "percentile@80": {
                "AUC@ROC": video_roc_auc.astype(float),
                "AUC@PR": video_pr_auc.astype(float),
                "Threshold": video_threshold_p.astype(float),
                "F1": video_f1_p.astype(float),
                "ACC": video_acc_p.astype(float),
                "precision": video_precision_p.astype(float),
                "recall": video_recall_p.astype(float),
                "TP": video_TP_p.astype(float),
                "FP": video_FP_p.astype(float),
                "TN": video_TN_p.astype(float),
                "FN": video_FN_p.astype(float),
                "TPR": video_TPR_p.astype(float),
                "FPR": video_FPR_p.astype(float)
            },
            # "best": {
            #     "AUC@ROC": video_roc_auc.astype(float),
            #     "AUC@PR": video_pr_auc.astype(float),
            #     "Threshold": video_threshold_b.astype(float),
            #     "F1": video_f1_b.astype(float),
            #     "ACC": video_acc_b.astype(float),
            #     "precision": video_precision_b.astype(float),
            #     "recall": video_recall_b.astype(float),
            #     "TP": video_TP_b.astype(float),
            #     "FP": video_FP_b.astype(float),
            #     "TN": video_TN_b.astype(float),
            #     "FN": video_FN_b.astype(float),
            #     "TPR": video_TPR_b.astype(float),
            #     "FPR": video_FPR_b.astype(float)
            # }
        }
    }

    return results, clip_scores, clip_labels


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






