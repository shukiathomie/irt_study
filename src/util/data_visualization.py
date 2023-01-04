import numpy as np
import matplotlib.pyplot as plt


class data_visualization:
    @classmethod
    def icc_show(cls, X, J, T):
        x = np.arange(1, 11)
        for j in range(5):
            y = X[j, :]
            plt.plot(x, y, label=j + 1)

        plt.title("Monotone Homogenity model ICC")
        plt.xlabel("latent abilities")
        plt.ylabel("probarility of correct answer")
        plt.legend()
        plt.show()
