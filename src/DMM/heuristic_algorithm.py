import sys
import numpy as np

sys.path.append("/Users/shukitakeuchi/irt_study/src")
from util.log import LoggerUtil
from joblib import Parallel, delayed
from tqdm import tqdm
from optimize_x import Opt_x
from optimize_y import Opt_y
from optimize_Z import Opt_Z
from optimize_W import Opt_W


class Heu_Algo:
    def __init__(self, U, Y, T):
        self.U = U
        self.init_Y = Y
        self.I, self.J = np.shape(self.U)
        self.T = T
        self.logger = LoggerUtil.get_logger(__name__)
        return

    def Parallel_step1(self, j):  # モデルの作成
        opt_x = Opt_x(self.U, self.init_Y, self.T)
        opt_x.modeling(j=j)
        # モデルの最適化
        x_opt, obj = opt_x.solve()
        return x_opt, obj

    def Parallel_step4(self, i, W_opt, Z_opt):
        # モデルの作成
        opt_y = Opt_y(self.U, W_opt, Z_opt, self.T)
        opt_y.modeling(i=i)
        # モデルの最適化
        y_opt, obj = opt_y.solve()
        return y_opt, obj

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
                delayed(Heu_Algo.Parallel_step1)(self, j) for j in range(self.J)
            )
        # self.logger.info(f"{out}")
        X_opt = np.concatenate([[sample[0]] for sample in out], axis=0)
        obj = np.concatenate([[sample[1]] for sample in out], axis=0)
        self.logger.info(f"X optimized ->{X_opt}")
        # step2
        self.logger.info("step2")
        opt_Z = Opt_Z(self.U, self.T)
        Z_opt = opt_Z.Est_Diff_Rank(X_opt)
        self.logger.info(f"Z optimized ->{Z_opt}")
        # step3
        self.logger.info("step3")
        opt_W = Opt_W(self.U, self.init_Y, Z_opt, self.T)
        opt_W.modeling()
        W_opt, obj = opt_W.solve()
        W_opt = np.reshape(W_opt, [self.J, self.T])
        self.logger.info(f"W optimized ->{W_opt}")
        # step4
        self.logger.info("step4")
        """Y_opt = np.empty((self.I, self.T), dtype=int)
        for i in range(self.I):
            opt_y = Opt_y(self.U, W_opt, Z_opt, self.T)
            opt_y.modeling(i=i)
            y_opt, obj = opt_y.solve()
            self.logger.info(f"y_{i}->{y_opt}")
            Y_opt[i, :] = y_opt"""
        # 並列化
        with LoggerUtil.tqdm_joblib(self.I):
            out = Parallel(n_jobs=-1, verbose=0)(
                delayed(Heu_Algo.Parallel_step4)(self, i, W_opt, Z_opt)
                for i in range(self.I)
            )
        # self.logger.info(f"{out}")
        Y_opt = np.concatenate([[sample[0]] for sample in out], axis=0)
        obj = np.concatenate([[sample[1]] for sample in out], axis=0)
        self.logger.info(f"Y optimized ->{Y_opt}")
        return W_opt, Y_opt, Z_opt
