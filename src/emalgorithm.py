from optimize import ProdPlan
from log import LoggerUtil
from pyomo.environ import *
from typing import Tuple
import os
import numpy as np
import itertools
from tqdm import tqdm
from joblib import Parallel, delayed


class EmAlgorithm:
    def __init__(self, U, Y, T, gamma) -> None:
        self.W = []
        # 初期化
        self.U = U
        self.Y = Y
        self.I, self.J = np.shape(self.U)
        self.T = T
        self.gamma = gamma
        self.W = np.zeros((self.J, self.T))
        self.a = [1] * self.J
        self.b = [1] * self.J
        self.a_list = np.arange(50, 210, 40) / 100
        self.b_list = np.arange(-150, 150, 40) / 100
        self.logger = LoggerUtil.get_logger(__name__)
        return

    @classmethod
    def fun_lambda(cls, Wjt, aj, bj) -> int:
        return 1 / (1 + np.exp(-1.7 * aj * (Wjt - bj)))

    @classmethod
    def con_prob(cls, X_jt, U_ij):
        return np.power(X_jt, U_ij) * np.power(1 - X_jt, 1 - U_ij)

    def convert_Y_calss(self, Y):
        index = np.argmax(Y, axis=1)
        Y = np.zeros((self.I, self.T))
        for i in range(len(index)):
            Y[i, index[i]] = 1
        return Y

    def EStep(self, pi, W, a, b):
        self.logger.info("EStep Start!")
        f = np.array(
            [
                [
                    np.prod(
                        [
                            EmAlgorithm.con_prob(
                                EmAlgorithm.fun_lambda(W[j, t], a[j], b[j]),
                                self.U[i, j],
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
        self.logger.info("Y:updated")
        self.logger.info("EStep finish")
        # self.logger.info(f"Y:{Y}")
        return Y

    def grid_search(self, j, Y):
        best_obj = np.inf
        best_W = None
        best_a = None
        best_b = None
        U_j = self.U[:, j : j + 1]
        for a, b in tqdm(itertools.product(self.a_list, self.b_list)):
            # モデルの作成
            plod_plan = ProdPlan(U_j, Y, a, b, T=self.T, gamma=self.gamma)
            plod_plan.modeling(j=j)
            # モデルの最適化
            # Wの最適化
            W_opt, objective = plod_plan.solve()
            if objective < best_obj:
                # self.logger.info(f"best_obj:{best_obj}->{objective}")
                best_obj = objective
                best_W = W_opt
                best_a = a
                best_b = b
            else:
                continue
        # self.logger.info(f"objective:{best_obj}")
        return best_W, best_a, best_b

    def MStep(self, Y, a, b):
        self.logger.info("MStep Start!")

        # piの更新
        pi = np.sum(Y, axis=0) / self.I
        self.logger.info("pi:updated")

        # Wの更新
        with LoggerUtil.tqdm_joblib(self.J):
            out = Parallel(n_jobs=-1, verbose=100)(
                delayed(EmAlgorithm.grid_search)(self, j, Y) for j in range(self.J)
            )
        best_W = np.concatenate([[np.array(sample[0])] for sample in out], axis=0)
        best_a = np.array([sample[1] for sample in out])
        best_b = np.array([sample[2] for sample in out])
        self.logger.info("W:updated")
        self.logger.info("MStep finish")
        self.logger.info(f"pi:{pi}")
        self.logger.info(f"W:{best_W}")
        self.logger.info(f"a:{best_a}")
        self.logger.info(f"b:{best_b}")
        return pi, best_W, best_a, best_b

    def repeat_process(self):
        # 初期ステップ -> MStep
        i = 1
        self.logger.info("first step")
        Y = self.Y
        a = self.a
        b = self.b
        pi, W, a, b = EmAlgorithm.MStep(self, Y, a, b)
        self.logger.info(f"pi:{pi}")
        self.logger.info(f"W:{W}")
        self.logger.info(f"a:{a}")
        self.logger.info(f"b:{b}")

        Yhat = np.empty((self.I, self.T))

        while np.any(Yhat != Y):
            # 繰り返し回数
            i += 1
            self.logger.info(f"{i}step")
            # EStep
            Y = EmAlgorithm.EStep(self, pi, W, a, b)
            # MStep
            pi, W, a, b = EmAlgorithm.MStep(self, Y, a, b)
            # 収束しない時、10回で終了させる
            if i == 20:
                return W, a, b, Y
        return W, a, b, Y
