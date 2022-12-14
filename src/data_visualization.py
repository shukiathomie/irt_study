import numpy as np
import matplotlib.pyplot as plt
from emalgorithm import EmAlgorithm


class data_visualization(EmAlgorithm):
    @classmethod
    def icc_show(cls, W, a, b, gamma):
        x = np.arange(1, 11)
        for j in range(5):
            y = 1 / (1 + np.exp(-1.7 * a[j] * (W[j] - b[j])))
            plt.plot(x, y, label=j + 1)

        plt.title("γ =  {} smooth-constraind model ICC".format(gamma))
        plt.xlabel("latent abilities")
        plt.ylabel("probarility of correct answer")
        plt.legend()
        plt.show()
