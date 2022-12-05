import numpy as np
import pandas as pd
import itertools
import sys

sys.path.append("/Users/shukitakeuchi/irt_study/src")
from util.log import LoggerUtil
from util.data_visualization import data_visualization
from sklearn.cluster import KMeans


class Clustering:
    def __init__(self, n_clusters, target, J):
        self.logger = LoggerUtil.get_logger(__name__)
        self.n_clusters = n_clusters
        self.target = target
        self.J = J

    def clustered(self):

        pred = KMeans(n_clusters=self.n_clusters, init="k-means++").fit_predict(
            self.target
        )
        # self.logger.info(f"pred:{pred}")
        target_df = pd.DataFrame(self.target)
        # self.logger.info(f"X_df:{target_df}")
        target_df["cluster_id"] = pred
        # self.logger.info(f"X_df:{target_df}")
        """data_visualization.cluster_icc(
            target_df, self.target, n_cluster=self.n_clusters, J=self.J
        )"""
        return target_df
