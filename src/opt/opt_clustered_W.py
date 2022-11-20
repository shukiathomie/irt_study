import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.environ import *


class Opt_W:
    def __init__(self, U, Y, Z, T, target_df, n_clusters) -> None:
        # モデル
        self.solver = "ipopt"
        # 初期設定
        self.U = U
        self.Y = Y
        self.Z = Z
        self.I, self.J = np.shape(self.U)
        self.target_df = target_df
        self.n_clusters = n_clusters
        self.T = T

    def modeling(self):
        # 非線形最適化モデル作成（minimize)
        self.model = pyo.ConcreteModel("Maximize Non Convex Optimization")
        # 変数のセット
        self.model.I = pyo.Set(initialize=range(1, self.I + 1))
        self.model.J = pyo.Set(initialize=range(1, self.J + 1))
        self.model.T = pyo.Set(initialize=range(1, self.T + 1))
        self.model.J1 = pyo.Set(initialize=range(1, self.J))
        self.model.T1 = pyo.Set(initialize=range(1, self.T))
        # 決定変数
        self.model.W = pyo.Var(
            self.model.J, self.model.T, domain=pyo.Reals, bounds=(0.01, 0.99)
        )
        # 制約
        self.model.const = pyo.ConstraintList()
        # 制約式
        # 制約1
        for k in self.model.J:
            for t in self.model.T1:
                lhs = self.model.W[k, t + 1] - self.model.W[k, t]
                self.model.const.add(lhs >= 0)
        # 制約2
        for n in range(self.n_clusters):
            cluster_list = []
            for j in range(self.J):
                if self.target_df["cluster_id"][j] == 1:
                    k = np.argmax(self.Z[j, :])
                    cluster_list.append(k + 1)
            cluster_list.sort()
            for t in self.model.T:
                for i in range(len(cluster_list) - 1):
                    lhs = (
                        self.model.W[cluster_list[i + 1], t]
                        - self.model.W[cluster_list[i], t]
                    )
                    self.model.const.add(lhs >= 0)

        # 目的関数
        """expr = sum(
            [
                [
                    [
                        [
                            (
                                self.Y[i - 1, t - 1]
                                * self.Z[j - 1, k - 1]
                                * (
                                    (self.U[i - 1, j - 1] * log(self.model.W[k, t]))
                                    + (
                                        (1 - self.U[i - 1, j - 1])
                                        * log(1 - self.model.W[k, t])
                                    )
                                )
                            )
                            for i in self.model.I
                        ]
                        for j in self.model.J
                    ]
                    for k in self.model.J
                ]
                for t in self.model.T
            ]
        )"""
        expr = sum(
            (
                self.Y[i - 1, t - 1]
                * self.Z[j - 1, k - 1]
                * (
                    (self.U[i - 1, j - 1] * log(self.model.W[k, t]))
                    + ((1 - self.U[i - 1, j - 1]) * log(1 - self.model.W[k, t]))
                )
            )
            for i in self.model.I
            for j in self.model.J
            for k in self.model.J
            for t in self.model.T
        )
        self.model.Obj = pyo.Objective(expr=expr, sense=pyo.maximize)
        # self.model.pprint()
        return

    def solve(self):
        opt = pyo.SolverFactory(self.solver)
        # opt.options["halt_on_ampl_error"] = "yes"
        opt.solve(self.model, tee=False)
        return pyo.value(self.model.W[:, :]), self.model.Obj()
