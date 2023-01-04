import sys
import numpy as np

sys.path.append("/Users/shukitakeuchi/irt_study/src")

from util.data_handling import data_handle
from util.data_visualization import data_visualization
from util.estimation_accuracy import est_accuracy
from util.log import LoggerUtil
from util.repo import repoUtil
from em_algorithm import MHM_EM_Algo


def main(T):
    logger = LoggerUtil.get_logger(__name__)
    # 実験の設定
    T = T
    # パスの指定
    indpath = "/Users/shukitakeuchi/Library/Mobile Documents/com~apple~CloudDocs/研究/項目反応理論/data/data0/30*1000"
    # データを読み込む
    U_df, Y_df, T_true_df, icc_true_df = data_handle.pandas_read(indpath)
    # nparrayに変換
    U, init_Y, T_true, icc_true, I, J = data_handle.df_to_array(
        U_df, Y_df, T_true_df, icc_true_df
    )

    mhm_em_algo = MHM_EM_Algo(U, init_Y, T)
    X_best, Y_best = mhm_em_algo.repeat_process()
    T_est = est_accuracy.show_class(Y_best)
    rsme_class = est_accuracy.rsme_class(T_true, T_est)
    logger.info(f"rsme_class:{rsme_class}")
    rmse_icc = est_accuracy.rmse_icc(icc_true, X_best)
    logger.info(f"rmse_icc:{rmse_icc}")
    return X_best, Y_best


if __name__ == "__main__":
    T = 10
    J = 30
    X_best, Y_best = main(T)
    # data_visualization.icc_show(X_best, J, T)
