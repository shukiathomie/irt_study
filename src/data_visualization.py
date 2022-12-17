import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # seabornをインポート

sns.set()


class data_visualization:
    @classmethod
    def icc_show(cls, W, Z, J, T):
        x = np.arange(1, 11)
        for j in range(J):
            y = [sum(W[k, t] * Z[j, k] for k in range(J)) for t in range(T)]
            plt.plot(x, y, label=j + 1)

        plt.title("Double monotonicity model ICC")
        plt.xlabel("latent abilities")
        plt.ylabel("probarility of correct answer")
        plt.legend()
        plt.show()
