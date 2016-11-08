# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import gammaln


class MultinomialDistribution(object):

    def __init__(self, p, rso=np.random):
        """初始化随机变量

        :param p: 事件概率
        :param rso: 随机数发生器
        """

        if not np.isclose(np.sum(p), 1.0):
            raise ValueError("event probalities do not sum to 1")

        self.p = p
        self.rso = rso
        self.logp = np.log(p)

    def sample(self, n):
        """从n个多随机变量分布事件中采样"""
        x = self.rso.multinomial(n, self.p)
        return x

    def log_pmf(self, x):
        """计算概率质量函数对数"""
        n = np.sum(x)
        log_n_factorial = gammaln(n + 1)
        sum_log_xi_factorial = np.sum(gammaln(x + 1))

        # 如果self.p中的某个值为0,那么logp=-inf, 0*inf=nan，所以要修正这一情况，让其为0
        log_pi_xi = self.logp * x
        log_pi_xi[x == 0] = 0

        sum_log_pi_xi = np.sum(log_pi_xi)

        log_pmf = log_n_factorial - sum_log_xi_factorial + sum_log_pi_xi
        return log_pmf

    def pmf(self, x):
        """计算概率质量函数pmf"""
        pmf = np.exp(self.log_pmf(x))
        return pmf

