# -*- coding: utf-8 -*-
from math import erf
import math
import numpy as np

_ERF = np.vectorize(erf, otypes=[float])


def norm_cdf(x):
    """
    Функция распределения стандартного нормального закона
    через функцию ошибок.
    """
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + _ERF(x / np.sqrt(2.0)))


def call_black_scholes(S, K, r, sigma, tau):
    """
    Цена европейского колл-опциона по формуле Блэка—Шоулза.

    Args:
        S: текущая цена базового актива.
        K: страйк.
        r: безрисковая ставка (годовая, используется 2%).
        sigma: годовая волатильность.
        tau: время до погашения в годах.
    """
    S = np.asarray(S, dtype=float)
    if tau <= 0 or sigma <= 0:
        return np.maximum(S - K, 0.0)
    sqrt_tau = math.sqrt(tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    Nd1 = norm_cdf(d1)
    Nd2 = norm_cdf(d2)
    return Nd1 * S - Nd2 * K * np.exp(-r * tau)
