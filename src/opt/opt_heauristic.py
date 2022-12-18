import sys

sys.path.append("/Users/shukitakeuchi/irt_study/src")

from util.log import LoggerUtil
from util.data_handling import data_handle
from util.estimation_accuracy import est_accuracy
from util.clustering import Clustering
from util.data_visualization import data_visualization
from joblib import Parallel, delayed
from tqdm import tqdm
from MHM.opt_MHM import Opt_MHM
from DMM.em_clustered_dmm_algo import EM_DMM_Algo
from DMM.optimize_Z import Opt_Z
from sklearn.cluster import KMeans


def main(T):
    logger = LoggerUtil.get_logger(__name__)
    # 実験の設定
    T = T
    # パスの指定
    indpath = "/Users/shukitakeuchi/Library/Mobile Documents/com~apple~CloudDocs/研究/項目反応理論/data/data0/30*100"
    # データを読み込む
    U_df, Y_df, T_true_df = data_handle.pandas_read(indpath)
    # nparrayに変換
    U, init_Y, T_true, I, J = data_handle.df_to_array(U_df, Y_df, T_true_df)
    n_clusters = 3
    # MHM(Y)
    opt_MHM = Opt_MHM(U, init_Y, T)
    X_opt = opt_MHM.opt()
    # 難易度行列Zを推定
    opt_Z = Opt_Z(U, T)
    Z_opt = opt_Z.Est_Diff_Rank(X_opt)
    # クラスタリング
    clustering = Clustering(n_clusters=n_clusters, target=X_opt, J=J)
    target_df = clustering.clustered()
    # DMM
    em_dmm_algo = EM_DMM_Algo(U, init_Y, T, target_df, n_clusters)
    W_opt, Y_opt = em_dmm_algo.repeat_process(Z_opt)
    T_est = est_accuracy.show_class(Y_opt)
    rsme_class = est_accuracy.rsme_class(T_true, T_est)
    logger.info(f"rsme_class:{rsme_class}")
    # logger.info(f"T_true:{T_true}")
    # logger.info(f"T_est:{T_est}")
    return W_opt, Y_opt, Z_opt


if __name__ == "__main__":
    T = 10
    J = 30
    W_best, Y_best, Z_best = main(T)
    data_visualization.DMM_icc_show(W_best, Z_best, J, T)
