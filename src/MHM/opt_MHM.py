import sys
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append("/Users/shukitakeuchi/irt_study/src")

from MHM.optimize_x import Opt_x
from util.log import LoggerUtil


class Opt_MHM:
    def __init__(self, U, init_Y, T):
        self.U = U
        self.init_Y = init_Y
        self.T = T
        self.I, self.J = np.shape(self.U)
        self.logger = LoggerUtil.get_logger(__name__)

    def Parallel_step(self, j):  # モデルの作成
        opt_x = Opt_x(self.U, self.init_Y, self.T)
        opt_x.modeling(j=j)
        # モデルの最適化
        x_opt, obj = opt_x.solve()
        return x_opt, obj

    def opt(self):
        self.logger.info("MHM(Y) start")
        """初期値Yを所与としてMHMについてXを解く"""
        with LoggerUtil.tqdm_joblib(self.J):
            out = Parallel(n_jobs=-1, verbose=0)(
                delayed(Opt_MHM.Parallel_step)(self, j) for j in range(self.J)
            )
        X_opt = np.concatenate([[sample[0]] for sample in out], axis=0)
        obj = np.concatenate([[sample[1]] for sample in out], axis=0)
        return X_opt
