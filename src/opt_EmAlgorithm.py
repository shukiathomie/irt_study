from data_handling import data_handle
from optimize import ProdPlan
from emalgorithm import EmAlgorithm
from data_visualization import data_visualization
from estimation_accuracy import est_accuracy
from log import LoggerUtil
from repo import repoUtil
import numpy as np


def main(T, gamma):
    logger = LoggerUtil.get_logger(__name__)
    # 実験の設定
    T = T
    gamma = gamma
    # パスの指定
    indpath = "/Users/shukitakeuchi/Library/Mobile Documents/com~apple~CloudDocs/研究/項目反応理論/data0/10*3000"
    outdpath = "/Users/shukitakeuchi/Library/Mobile Documents/com~apple~CloudDocs/研究/項目反応理論/NonPLmodel/output"
    # データを読み込む
    U_df, Y_df, T_true_df = data_handle.pandas_read(indpath)
    # nparrayに変換
    U, Y, T_true = data_handle.df_to_array(U_df, Y_df, T_true_df)
    # EmAlgorithm
    EM = EmAlgorithm(U, Y, T, gamma)
    W, a, b, Y_est = EM.repeat_process()
    T_est = est_accuracy.show_class(Y_est)
    repoUtil.output_csv(outdpath, T_est, "T_est")
    rsme_class = est_accuracy.rsme_class(T_true, T_est)
    logger.info(f"rsme_class:{rsme_class}")
    # logger.info(f"T_true:{T_true}")
    # logger.info(f"T_est:{T_est}")
    return W, a, b, Y_est


if __name__ == "__main__":
    gamma = 0
    W, a, b, Y = main(10, gamma)
    data_visualization.icc_show(W, a, b, gamma)
