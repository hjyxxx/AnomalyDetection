import numpy as np


class Graph:
    def __init__(self, layout, strategy, pose_num=1, seg_len=1):
        """
        生成邻接矩阵
        :param layout: 姿态点布局
        :param strategy: 生成邻接矩阵的策略
        :param pose_num: 姿态点个数
        :param seg_len: 序列长度
        """
        self.pose_num = pose_num
        self.seg_len = seg_len
        self.get_pose_edge(layout)
        self.get_adjacency(strategy)

    def get_pose_edge(self, layout):
        """
        获得边
        :param layout:
        :return:
        """
        if layout == 'openpose':
            self.pose_num = 18
            self_link = [(i, i) for i in range(self.pose_num)]
            neighbor_link = [(15, 0), (14, 0), (17, 15), (16, 14),
                             (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 1)]

            self.pose_edge = self_link + neighbor_link

        elif layout == 'asd':
            self_link = [(i, i) for i in range(self.pose_num)]

            neighbor_link = []
            for i in range(self.pose_num):
                for j in range(self.pose_num):
                    if i != j and (i, j) not in neighbor_link and (j, i) not in neighbor_link:
                        neighbor_link.append((i, j))

            self.pose_edge = self_link + neighbor_link

        else:
            raise ValueError("Do Not Exist This layout: {}".format(layout))

    def get_adjacency(self, strategy):
        pose_A = np.zeros((self.pose_num, self.pose_num))       # 姿态点的邻接矩阵
        for i, j in self.pose_edge:
            pose_A[i, j] = 1
            pose_A[j, i] = 1

        if strategy == 'uniform':
            self.node_num = self.pose_num
            A = pose_A
            self.A = self.normalize_digraph(A)

        elif strategy == 'NaturalConnection':
            self.node_num = self.pose_num * self.seg_len
            A = np.zeros((self.node_num, self.node_num))      # (VL, VL)
            for i in range(self.seg_len):
                for j in range(self.seg_len):
                    row = i * self.pose_num
                    col = j * self.pose_num
                    if i == j:
                        A[row: row + self.pose_num, col: col + self.pose_num] = pose_A
                    else:
                        A[row: row + self.pose_num, col: col + self.pose_num] = np.eye(self.pose_num)

            self.A = self.normalize_digraph(A)

        elif strategy == 'TimeFullConnect':

            self.node_num = self.pose_num * self.seg_len

            A = np.zeros((self.node_num, self.node_num))      # (VL, VL)

            for i in range(self.seg_len):
                for j in range(self.seg_len):
                    row = i * self.pose_num
                    col = j * self.pose_num

                    if i == j:
                        A[row: row + self.pose_num, col: col + self.pose_num] = pose_A
                    else:
                        A[row: row + self.pose_num, col: col + self.pose_num] = np.ones((self.pose_num, self.pose_num))

            self.A = self.normalize_digraph(A)

        elif strategy == 'FullConnect':
            self.node_num = self.pose_num * self.seg_len
            adj = np.ones((self.node_num, self.node_num))
            self.A = self.normalize_digraph(adj)

        else:
            raise ValueError("Do Not Exist This strategy: {}".format(strategy))

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        node_num = A.shape[0]
        Dn = np.zeros((node_num, node_num))
        for i in range(node_num):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)

        AD = np.dot(A, Dn)

        return AD


if __name__ == '__main__':

    graph = Graph('openpose', 'pure', 12)

    print(graph.A)