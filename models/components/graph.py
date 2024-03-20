import numpy as np


class Graph:
    def __init__(self, layout, strategy, seg_len):
        self.seg_len = seg_len
        self.get_edge(layout)
        self.get_adjacency(strategy)

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(15, 0), (14, 0), (17, 15), (16, 14),
                             (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 1)]

            self.edge = self_link + neighbor_link
        else:
            raise ValueError("Do Not Exist This layout: {}".format(layout))

    def get_adjacency(self, strategy):
        pose_A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            pose_A[i, j] = 1
            pose_A[j, i] = 1
        if strategy == 'uniform':
            self.A = self.normalize_digraph(pose_A)
        elif strategy == 'pure':
            A = np.zeros((self.num_node * self.seg_len, self.num_node * self.seg_len))      # (VL, VL)
            for i in range(self.seg_len):
                for j in range(self.seg_len):

                    row = i * self.num_node
                    col = j * self.num_node
                    if i == j:
                        A[row: row + self.num_node, col: col + self.num_node] = pose_A
                    else:
                        A[row: row + self.num_node, col: col + self.num_node] = np.ones((self.num_node, self.num_node))

            self.A = self.normalize_digraph(A)

        elif strategy == 'full':
            adj = np.ones((self.num_node * self.seg_len, self.num_node * self.seg_len))
            self.A = self.normalize_digraph(adj)

        else:
            raise ValueError("Do Not Exist This strategy: {}".format(strategy))

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)

        AD = np.dot(A, Dn)

        return AD


if __name__ == '__main__':

    graph = Graph('openpose', 'pure', 12)

    print(graph.A)