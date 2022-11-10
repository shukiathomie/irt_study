import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.environ import *


class ProdPlan:
    def __init__(self, U_j, Y, a, b, T, gamma) -> None:
        # モデル
        self.solver = "ipopt"
        # 初期設定
        self.U_j = U_j
        self.Y = Y
        self.aj = a
        self.bj = b
        self.T = T
        self.gamma = gamma
        self.I = len(self.U_j)

    def modeling(self, j):
        # 非線形最適化モデル作成（minimize)
        self.model = pyo.ConcreteModel("Minimize Non Convex Optimization")
        # 変数のセット
        self.model.I = pyo.Set(initialize=range(1, self.I + 1))
        self.model.T = pyo.Set(initialize=range(1, self.T + 1))
        # 決定変数
        self.model.W_j = pyo.Var(self.model.T, domain=pyo.Reals)
        self.model.S_j = pyo.Var(
            self.model.T, domain=pyo.Reals, bounds=(0, float("inf"))
        )
        self.model.V_j = pyo.Var(
            self.model.T, domain=pyo.Reals, bounds=(0, float("inf"))
        )
        # 制約
        self.model.const = pyo.ConstraintList()
        # 制約式
        # 制約1
        for t in range(1, self.T):
            lhs = self.model.W_j[t + 1] - self.model.W_j[t]
            self.model.const.add(lhs >= 0)
        # 制約2
        const2 = (
            sum(self.model.S_j[t] + self.model.V_j[t] for t in self.model.T)
            - self.gamma
        )
        self.model.const.add(const2 <= 0)
        # 制約3
        for t in range(1, self.T - 1):
            const3_lhs = (
                self.model.S_j[t]
                - self.model.V_j[t]
                - self.model.W_j[t + 2]
                + 2 * self.model.W_j[t + 1]
                - self.model.W_j[t]
            )
            self.model.const.add(const3_lhs == 0)
        # 目的関数
        # expr = np.sum(np.multiply(self.Y, np.log(1+np.exp(np.dot(1-2*self.U_j,(-1.7*self.a[j]*(self.model.W_j-self.b[j])))))))
        expr = sum(
            self.Y[i - 1, t - 1]
            * log(
                1
                + exp(
                    (1 - 2 * self.U_j[i - 1])
                    * 1.7
                    * self.aj
                    * (self.model.W_j[t] - self.bj)
                )
            )
            for i in self.model.I
            for t in self.model.T
        )
        self.model.Obj = pyo.Objective(expr=expr, sense=pyo.minimize)
        return

    def solve(self):
        opt = pyo.SolverFactory(self.solver)
        # opt.options["halt_on_ampl_error"] = "yes"
        res = opt.solve(self.model, tee=False)
        return pyo.value(self.model.W_j[:]), self.model.Obj()

    def show_input(self):
        return

    def show_model(self):
        return

    def show_result(self):
        return
