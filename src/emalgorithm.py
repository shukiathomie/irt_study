from optimize import ProdPlan
import os
import numpy as np
from pyomo.environ import *
from log import LoggerUtil
from joblib import Parallel, delayed
from typing import Tuple
from tqdm import tqdm


class EmAlgorithm:
    def __init__(self, U, Y, T, gamma):
        # 変数の定義
        self.U = U
        self.Y = Y
        self.I, self.J = np.shape(self.U)
        self.T = T
        self.gamma = gamma
        self.logger = LoggerUtil.get_logger(__name__)
        return

    @classmethod
    def fun_lambda(cls, W) -> np.array:
        return 1 / (1 + np.exp(-W))

    @classmethod
    def con_prob(cls, X_jt, U_ij):
        return np.power(X_jt, U_ij) * np.power(1 - X_jt, 1 - U_ij)

    def convert_Y_calss(self, Y):
        index = np.argmax(Y, axis=1)
        Y = np.zeros((self.I, self.T), dtype=int)
        for i in range(len(index)):
            Y[i, index[i]] = 1
        return Y

    def EStep(self, pi, W):

        self.logger.info("EStep start")
        f = np.array(
            [
                [
                    np.prod(
                        [
                            EmAlgorithm.con_prob(
                                EmAlgorithm.fun_lambda(W[j, t]), self.U[i, j]
                            )
                            for j in range(self.J)
                        ]
                    )
                    for t in range(self.T)
                ]
                for i in range(self.I)
            ]
        )
        f1 = pi * f
        f2 = np.sum(f1, 1).reshape(-1, 1)
        f3 = f1 / f2
        Y = EmAlgorithm.convert_Y_calss(self, f3)
        self.logger.info("EStep finish")
        # self.logger.info(f"Y:{Y}")
        return Y

    def Parallel_processing(self, Y, j):
        # モデルの作成
        # Uをjで分割
        U_j = self.U[:, j : j + 1]
        plod_plan = ProdPlan(U_j, Y, self.T, self.gamma)
        plod_plan.modeling(j=j)
        # モデルの最適化
        W_opt, obj = plod_plan.solve()
        return W_opt, obj

    def MStep(self, Y):
        self.logger.info("MStep start")
        # piの更新
        pi = np.sum(Y, axis=0) / self.I

        # Wの更新
        with LoggerUtil.tqdm_joblib(self.J):
            out = Parallel(n_jobs=-1, verbose=0)(
                delayed(EmAlgorithm.Parallel_processing)(self, Y, j)
                for j in range(self.J)
            )
        # self.logger.info(f"{out}")
        W = np.concatenate([[sample[0]] for sample in out], axis=0)
        obj = np.concatenate([[sample[1]] for sample in out], axis=0)
        self.logger.info("MStep finish")
        # self.logger.info(f"objective:{obj}")
        return pi, W

    def repeat_process(self):
        # 初期ステップ -> MStep
        i = 1
        # Yを初期化
        Y = self.Y
        self.logger.info("first step")
        pi, W = EmAlgorithm.MStep(self, Y)
        est_Y = np.empty((self.I, self.T))
        while np.any(est_Y != Y):
            est_Y = Y
            # 繰り返し回数
            i += 1
            self.logger.info(f"{i}th step")
            # EStep
            Y = EmAlgorithm.EStep(self, pi, W)
            # MStep
            pi, W = EmAlgorithm.MStep(self, Y)
            # 収束しない時、50回で終了させる
            if i == 20:
                return W, Y
        return W, Y
