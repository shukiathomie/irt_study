import sys
import numpy as np

sys.path.append("/Users/shukitakeuchi/irt_study/src")
from util.log import LoggerUtil
from joblib import Parallel, delayed
from tqdm import tqdm
from DMM.optimize_x import Opt_x
from DMM.optimize_Z import Opt_Z
from opt.opt_clustered_W import Opt_W


class EM_DMM_Algo:
    def __init__(self, U, Y, T, target_df, n_clusters):
        self.U = U
        self.init_Y = Y
        self.I, self.J = np.shape(self.U)
        self.T = T
        self.target_df = target_df
        self.n_clusters = n_clusters
        self.logger = LoggerUtil.get_logger(__name__)
        return

    @classmethod
    def con_prob(cls, W_kt, Z_jk, U_ij):
        return np.power(np.power(W_kt, U_ij) * np.power(1 - W_kt, 1 - U_ij), Z_jk)

    def convert_Y_calss(self, Y):
        index = np.argmax(Y, axis=1)
        Y = np.zeros((self.I, self.T), dtype=int)
        for i in range(len(index)):
            Y[i, index[i]] = 1
        return Y

    def Parallel_step1(self, j):  # モデルの作成
        opt_x = Opt_x(self.U, self.init_Y, self.T)
        opt_x.modeling(j=j)
        # モデルの最適化
        x_opt, obj = opt_x.solve()
        return x_opt, obj

    def EStep(self, pi, W, Z):

        self.logger.info("EStep start")
        f = np.array(
            [
                [
                    np.prod(
                        [
                            EM_DMM_Algo.con_prob(W[k, t], Z[j, k], self.U[i, j])
                            for j in range(self.J)
                            for k in range(self.J)
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
        Y = EM_DMM_Algo.convert_Y_calss(self, f3)
        self.logger.info("EStep finish")
        # self.logger.info(f"Y:{Y}")
        return Y

    def MStep(self, Y, Z_opt):
        self.logger.info("MStep start")
        # piの更新
        pi = np.sum(Y, axis=0) / self.I

        # Wの更新
        opt_W = Opt_W(
            self.U, self.init_Y, Z_opt, self.T, self.target_df, self.n_clusters
        )
        opt_W.modeling()
        W_opt, obj = opt_W.solve()
        W_opt = np.reshape(W_opt, [self.J, self.T])
        self.logger.info(f"W optimized ->{W_opt}")
        self.logger.info("MStep finish")
        # self.logger.info(f"objective:{obj}")
        return pi, W_opt

    def repeat_process(self, Z_opt):
        # 初期ステップ -> MStep
        i = 1
        # Yを初期化
        Y = self.init_Y
        self.logger.info("first step")
        pi, W = EM_DMM_Algo.MStep(self, Y, Z_opt)
        est_Y = np.empty((self.I, self.T))
        while np.any(est_Y != Y):
            est_Y = Y
            # 繰り返し回数
            i += 1
            self.logger.info(f"{i}th step")
            # EStep
            Y = EM_DMM_Algo.EStep(self, pi, W, Z_opt)
            # MStep
            pi, W = EM_DMM_Algo.MStep(self, Y, Z_opt)
            # 収束しない時、50回で終了させる
            if i == 5:
                return W, Y
        return W, Y

    def process(self):
        # step0 init_Y
        # step1
        self.logger.info("step1")
        """X_opt = np.empty((self.J, self.T))
        for j in range(self.J):
            self.logger.info(f"{j+1}th item optimized")
            opt_x = Opt_x(self.U, self.init_Y, self.T)
            opt_x.modeling(j=j)
            x_opt, obj = opt_x.solve()
            # self.logger.info(f"x_{j+1}->{x_opt}")
            X_opt[j, :] = x_opt"""
        # 並列化
        with LoggerUtil.tqdm_joblib(self.J):
            out = Parallel(n_jobs=-1, verbose=0)(
                delayed(EM_DMM_Algo.Parallel_step1)(self, j) for j in range(self.J)
            )
        # self.logger.info(f"{out}")
        X_opt = np.concatenate([[sample[0]] for sample in out], axis=0)
        obj = np.concatenate([[sample[1]] for sample in out], axis=0)
        # self.logger.info(f"X optimized ->{X_opt}")
        # step2
        self.logger.info("step2")
        opt_Z = Opt_Z(self.U, self.T)
        Z_opt = opt_Z.Est_Diff_Rank(X_opt)
        # self.logger.info(f"Z optimized ->{Z_opt}")
        # emstep
        self.logger.info("emstep")
        W_opt, Y_opt = EM_DMM_Algo.repeat_process(self, Z_opt)
        return W_opt, Y_opt, Z_opt
