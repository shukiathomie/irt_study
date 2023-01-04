import numpy as np


class est_accuracy:
    @classmethod
    def show_class(cls, Y):
        T = 1 + np.argmax(Y, 1)
        return T

    @classmethod
    def rsme_class(cls, T_true, T_est):
        I = len(T_true)
        rsme = np.sqrt(
            np.sum([np.square(T_true[i] - T_est[i]) for i in range(len(T_true))]) / I
        )
        return rsme

    @classmethod
    def rmse_icc(cls, icc_true, X):
        J, T = np.shape(X)
        rmse = np.sqrt(
            np.sum(
                np.square(icc_true[j, t] - X[j, t]) for j in range(J) for t in range(T)
            )
            / (J * T)
        )
        return rmse
