"""
A basic SVD implementation. Done with the idea that this will help
me understand the method more broadly.
"""

import numpy as np


class SVDBase:
    def __init__(self):
        self._reset()

    def _reset(self):
        self.U = None
        self.V = None
        self.row_map = {}
        self.col_map = {}
        self.m = 0
        self.n = 0

    def _init_row_map(self, data):
        userIds = sorted(set([d[0] for d in data]))
        for i, userId in enumerate(userIds):
            self.row_map[userId] = i
        self.m = len(self.row_map.keys())

    def _init_col_map(self, data):
        itemIds = sorted(set([d[1] for d in data]))
        for i, itemId in enumerate(itemIds):
            self.col_map[itemId] = i
        self.n = len(self.col_map.keys())

    def learn(self, data, dim, rate, reg_pen, tran_pen, max_its):
        """Learn a predictor based on data tuples.

        data: a list of 3 tuples (userId, itemId, rating)
        rate: the learning rate
        dim: the dimension to factor into.
        """
        self._reset()
        self._init_row_map(data)
        self._init_col_map(data)

        U = np.random.rand(self.m, dim)
        V = np.random.rand(dim, self.n)

        self.U, self.V = self.sgd(data, U, V, rate, reg_pen, tran_pen, max_its)
        return self

    def sgd(self, data, U, V, rate, reg_pen, tran_pen, max_its):
        for i in range(max_its):
            for userId, itemId, rating in data:
                # Get the correct row and column, dot them for prediction
                i, j = self.row_map[userId], self.col_map[itemId]
                u_i = U[i, :]
                v_j = V[:, j]
                pred = np.dot(u_i, v_j)

                # Caclulate error
                err = rating - pred

                # Calculate new vectors
                u_i_new = u_i + rate * (err * v_j - reg_pen * u_i)
                v_j_new = v_j + rate * (err * u_i - reg_pen * v_j)

                # Set new vectors
                U[i, :] = u_i_new
                V[:, j] = v_j_new

        return U, V

    def predict(self, userId, itemId):
        i, j = self.row_map[userId], self.col_map[itemId]
        u_i = self.U[i, :]
        v_j = self.V[:, j]
        return np.dot(u_i, v_j)

    def prediction_matrix(self):
        return np.matmul(self.U, self.V)
