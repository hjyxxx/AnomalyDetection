import math

import numpy as np
import torch


def get_aff_trans_mat(sx=1, sy=1, rot=0, flip=False):
    """
    Generate affine transfomation matrix (torch.tensor type) for transforming pose sequences
    :rot is given in degrees
    """
    cos_r = math.cos(math.radians(rot))
    sin_r = math.sin(math.radians(rot))
    flip_mat = torch.eye(2, dtype=torch.float32)
    if flip:
        flip_mat[0, 0] = -1.0
    trans_scale_mat = torch.tensor([[sx, 0], [0, sy]], dtype=torch.float32)
    rot_mat = torch.tensor([[cos_r, -sin_r], [sin_r, cos_r]], dtype=torch.float32)
    aff_mat = torch.matmul(rot_mat, trans_scale_mat)
    aff_mat = torch.matmul(flip_mat, aff_mat)
    return aff_mat


def apply_pose_transform(pose, trans_mat):
    """
    Given a set of pose sequences of shape (Channels, Time_steps, Vertices, M[=num of figures])
    return its transformed form of the same sequence. 3 Channels are assumed (x, y, conf)
    :param pose: (C, L, V)
    """
    if len(pose.shape) == 3:
        einsum_str = 'ktv,ck->ctv'
    else:
        einsum_str = 'ktvm,ck->ctvm'
    pose_transformed = np.einsum(einsum_str, pose, trans_mat)
    return pose_transformed


class PoseTransform(object):
    """ A general class for applying transformations to pose sequences, empty init returns identity """

    def __init__(self, sx=1, sy=1, rot=0, flip=False, trans_mat=None):
        """ An explicit matrix overrides all parameters"""
        if trans_mat is not None:
            self.trans_mat = trans_mat
        else:
            self.trans_mat = get_aff_trans_mat(sx, sy, rot, flip)

    def __call__(self, x):
        x = apply_pose_transform(x, self.trans_mat)
        return x

ae_trans_list = [
    PoseTransform(sx=1, sy=1, rot=0, flip=False),  # 0
    PoseTransform(sx=1, sy=1, rot=0, flip=True),  # 3
    PoseTransform(sx=1, sy=1, rot=90, flip=False),  # 6
    PoseTransform(sx=1, sy=1, rot=90, flip=True),  # 9
    PoseTransform(sx=1, sy=1, rot=45, flip=False),  # 12
]