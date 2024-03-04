import torch


def rescaled_L(L):
    """
    L_tilde = 2*L/lambda_max - I
    对Laplacian矩阵进行缩放到[-1, 1]
    :param L: tensor, (v, v)
    :return:
    """
    v, v = L.shape
    I = torch.eye(v)

    # 计算特征值
    eigenvalues, _ = torch.linalg.eig(L)

    # 找到最大特征值
    lambda_max = eigenvalues.real.max()

    L_tilde = (2 * L) / lambda_max - I

    return L_tilde
