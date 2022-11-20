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
    indpath = "/Users/shukitakeuchi/Library/Mobile Documents/com~apple~CloudDocs/研究/項目反応理論/data0/30*3000"
    outdpath = "/Users/shukitakeuchi/Library/Mobile Documents/com~apple~CloudDocs/研究/項目反応理論/NonPLmodel/output"
    # データを読み込む
    U_df, Y_df, T_true_df = data_handle.pandas_read(indpath)
    # nparrayに変換
    U, Y, T_true = data_handle.df_to_array(U_df, Y_df, T_true_df)
    # Heuristic Algorithm
    heu_algo = Heu_Algo(U, Y, T)
    X_best, Y_best = heu_algo.repeat_process(Y)
    T_est = est_accuracy.show_class(Y_best)
    # repoUtil.output_csv(outdpath, T_est, "T_est")
    rsme_class = est_accuracy.rsme_class(T_true, T_est)
    logger.info(f"rsme_class:{rsme_class}")
    # logger.info(f"T_true:{T_true}")
    # logger.info(f"T_est:{T_est}")
    return X_best, Y_best


if __name__ == "__main__":
    T = 10
    J = 30
    X_best, Y_best = main(T)
    data_visualization.icc_show(X_best, J, T)
