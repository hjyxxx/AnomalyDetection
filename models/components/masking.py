import torch


class TriangularCausalMask():
    def __init__(self, L, S, device="cpu"):
        """
        :param L: 批次大小
        :param S: 注意力节点个数
        :param device:
        """
        mask_shape = [L, S]
        with torch.no_grad():
            # 生成上三角矩阵
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
