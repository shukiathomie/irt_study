import sys
import pandas as pd

sys.path.append("/Users/shukitakeuchi/irt_study/src")

from util.log import LoggerUtil
from util.data_handling import data_handle
from util.estimation_accuracy import est_accuracy

# from util.repo import repoUtil
from util.clustering import Clustering
from util.data_visualization import data_visualization
from MHM.heuristic_algorithm import Heu_MHM_Algo
from opt.clustered_dmm_algo import Heu_DMM_Algo
from sklearn.cluster import KMeans


def main(T):
    logger = LoggerUtil.get_logger(__name__)
    # 実験の設定
    T = T
    # パスの指定
    indpath = "/Users/shukitakeuchi/Library/Mobile Documents/com~apple~CloudDocs/研究/項目反応理論/data0/60*100"
    outdpath = "/Users/shukitakeuchi/Library/Mobile Documents/com~apple~CloudDocs/研究/項目反応理論/NonPLmodel/output"
    # データを読み込む
    U_df, Y_df, T_true_df = data_handle.pandas_read(indpath)
    # nparrayに変換
    U, Y, T_true, I, J = data_handle.df_to_array(U_df, Y_df, T_true_df)
    # Heuristic Algorithm
    logger.info("MHM start")
    heu_mhm_algo = Heu_MHM_Algo(U, Y, T)
    X_best, Y_best = heu_mhm_algo.repeat_process(Y)
    T_est = est_accuracy.show_class(Y_best)
    # ->repoUtil.output_csv(outdpath, T_est, "T_est")
    rsme_class = est_accuracy.rsme_class(T_true, T_est)
    logger.info(f"rsme_class:{rsme_class}")
    logger.info("MHM finish")
    # logger.info(f"T_true:{T_true}")
    # logger.info(f"T_est:{T_est}")
    n_clusters = 3
    clustering = Clustering(n_clusters=n_clusters, target=X_best, J=J)
    target_df = clustering.clustered()
    logger.info("DMM start")
    heu_dmm_algo = Heu_DMM_Algo(U, Y, T, target_df, n_clusters)
    W_est, Y_est, Z_est = heu_dmm_algo.process()
    T_est = est_accuracy.show_class(Y_est)
    # repoUtil.output_csv(outdpath, T_est, "T_est")
    rsme_class = est_accuracy.rsme_class(T_true, T_est)
    logger.info(f"rsme_class:{rsme_class}")
    logger.info("DMM finish")
    # logger.info(f"T_true:{T_true}")
    # logger.info(f"T_est:{T_est}")
    return W_est, Y_est, Z_est


if __name__ == "__main__":
    T = 10
    J = 30
    W_best, Y_best, Z_best = main(T)
    data_visualization.DMM_icc_show(W_best, Z_best, J, T)
