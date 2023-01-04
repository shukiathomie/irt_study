import sys
import numpy as np

sys.path.append("/Users/shukitakeuchi/irt_study/src")

from util.log import LoggerUtil
from joblib import Parallel, delayed
from tqdm import tqdm
from optimize_x import Opt_x


class MHM_EM_Algo:
    def __init__(self, U, Y, T):
        self.U = U
        self.init_Y = Y
        self.I, self.J = np.shape(self.U)
        self.T = T
        self.logger = LoggerUtil.get_logger(__name__)
        return

    @classmethod
    def con_prob(cls, X_jt, U_ij):
        return np.multiply(np.power(X_jt, U_ij), np.power(1 - X_jt, 1 - U_ij))

    def convert_Y_calss(self, Y):
        index = np.argmax(Y, axis=1)
        Y = np.zeros((self.I, self.T), dtype=int)
        for i in range(len(index)):
            Y[i, index[i]] = 1
        return Y

    # 適当
    def EStep(self, pi, X):
        f = np.array(
            [
                [
                    np.prod(
                        [
                            MHM_EM_Algo.con_prob(X[j, t], self.U[i, j])
                            for j in range(self.J)
                        ]
                    )
                    for t in range(self.T)
                ]
                for i in range(self.I)
            ]
        )
        f1 = np.multiply(pi, f)
        f2 = np.sum(f1, 1).reshape(-1, 1)
        Y = np.divide(f1, f2)
        Y_opt = MHM_EM_Algo.convert_Y_calss(self, Y)
        return Y, Y_opt

    def Parallel_step(self, j, Y):  # モデルの作成
        opt_x = Opt_x(self.U, Y, self.T)
        opt_x.modeling(j=j)
        # モデルの最適化
        x_opt, obj = opt_x.solve()
        return x_opt, obj

    def MStep(self, Y):
        # piの更新
        pi = np.sum(Y, axis=0) / self.I

        """X_opt = np.empty((self.J, self.T))
        for j in range(self.J):
            # self.logger.info(f"{j+1}th item optimized")
            opt_x = Opt_x(self.U, Y, self.T)
            opt_x.modeling(j=j)
            x_opt, obj = opt_x.solve()
            # self.logger.info(f"x_{j+1}->{x_opt}")
            X_opt[j, :] = x_opt"""

        with LoggerUtil.tqdm_joblib(self.J):
            out = Parallel(n_jobs=-1, verbose=0)(
                delayed(MHM_EM_Algo.Parallel_step)(self, j, Y) for j in range(self.J)
            )
        X_opt = np.concatenate([[sample[0]] for sample in out], axis=0)
        obj = np.concatenate([[sample[1]] for sample in out], axis=0)
        return pi, X_opt

    def repeat_process(self):
        # 初期ステップ -> MStep
        i = 1
        # Yを初期化
        Y_opt = self.init_Y
        self.logger.info("first step")
        pi, X = MHM_EM_Algo.MStep(self, Y_opt)
        est_Y = np.empty((self.I, self.T))
        while np.any(est_Y != Y_opt):
            est_Y = Y_opt
            # 繰り返し回数
            i += 1
            self.logger.info(f"{i}th step")
            # EStep
            self.logger.info(f"E-STEP")
            Y, Y_opt = MHM_EM_Algo.EStep(self, pi, X)
            # MStep
            self.logger.info(f"M-STEP")
            pi, X = MHM_EM_Algo.MStep(self, Y)
            # 収束しない時、30回で終了させる
            if i == 30:
                return X, Y_opt
        return X, Y_opt
