from data_handling import data_handle
from heuristic_algorithm import Heu_Algo
from data_visualization import data_visualization
from estimation_accuracy import est_accuracy
from log import LoggerUtil
from repo import repoUtil


def main(T):
    logger = LoggerUtil.get_logger(__name__)
    # 実験の設定
    T = T
    # パスの指定
    indpath = "/Users/shukitakeuchi/Library/Mobile Documents/com~apple~CloudDocs/研究/項目反応理論/data9/30*100"
    outdpath = "/Users/shukitakeuchi/Library/Mobile Documents/com~apple~CloudDocs/研究/項目反応理論/NonPLmodel/output"
    # データを読み込む
    U_df, Y_df, T_true_df = data_handle.pandas_read(indpath)
    # nparrayに変換
    U, Y, T_true = data_handle.df_to_array(U_df, Y_df, T_true_df)
    # EmAlgorithm
    heu_algo = Heu_Algo(U, Y, T)
    W_est, Y_est, Z_est = heu_algo.process()
    T_est = est_accuracy.show_class(Y_est)
    # repoUtil.output_csv(outdpath, T_est, "T_est")
    rsme_class = est_accuracy.rsme_class(T_true, T_est)
    logger.info(f"rsme_class:{rsme_class}")
    # logger.info(f"T_true:{T_true}")
    # logger.info(f"T_est:{T_est}")
    return W_est, Y_est, Z_est


if __name__ == "__main__":
    T = 10
    J = 30
    W_est, Y_est, Z_est = main(T)
    # data_visualization.icc_show(W_est, Z_est, J, T)
