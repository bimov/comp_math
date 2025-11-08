"""Численное решение уравнения Дюпира методом Крэнка—Николсона."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable
import numpy as np


@dataclass
class DupireSolution:
    """Хранит рассчитанную поверхность опциона на сетке (S, tau)."""

    S_grid: np.ndarray
    tau_grid: np.ndarray
    values: Dict[float, np.ndarray]

    def get_values(self, tau: float, tol: float = 1e-10) -> np.ndarray:
        """Вернуть значения C(S_i, tau)."""

        key = _round_tau(tau)
        if key in self.values:
            return self.values[key]
          
        for tau_key, vec in self.values.items():
            if abs(tau_key - tau) <= tol:
                return vec
        raise KeyError(f"Tau={tau:.6f} отсутствует в рассчитанной сетке")


def solve_dupire_crank_nicolson(S_grid: Iterable[float], tau_nodes: Iterable[float], sigma_nodes: Iterable[float], r: float, K: float) -> DupireSolution:
    """Решить уравнение Дюпира на сетке (S, tau) методом Крэнка—Николсона."""

    S = np.asarray(S_grid, dtype=float)
    if S.ndim != 1 or len(S) < 3:
        raise ValueError("Сетка S должна содержать минимум 3 узла")

    tau = np.asarray(list(tau_nodes), dtype=float)
    sigma = np.asarray(list(sigma_nodes), dtype=float)
    if len(tau) != len(sigma):
        raise ValueError("Размеры tau_nodes и sigma_nodes должны совпадать")
    if tau[0] != 0.0:
        raise ValueError("Первая точка по времени должна быть tau=0")
    if not np.all(np.diff(tau) >= 0):
        raise ValueError("Сетка tau должна быть неубывающей")

    dS = float(S[1] - S[0])
    if not np.allclose(np.diff(S), dS, atol=1e-12):
        raise ValueError("Сетка S предполагается равномерной")

    N = len(S)
    payoff = np.maximum(S - K, 0.0)
    values: Dict[float, np.ndarray] = {}

    V_prev = payoff.copy()
    V_prev[0] = 0.0
    values[_round_tau(0.0)] = V_prev.copy()

    for n in range(len(tau) - 1):
        tau_curr = tau[n]
        tau_next = tau[n + 1]
        dt = tau_next - tau_curr
        if dt <= 0:
            continue

        sigma_step = max(float(sigma[n + 1]), 1e-8)
        V_next = _crank_nicolson_step(V_prev, dt=dt, sigma=sigma_step, S=S, dS=dS, r=r, K=K, tau_next=tau_next)
        values[_round_tau(tau_next)] = V_next.copy()
        V_prev = V_next

    return DupireSolution(S_grid=S.copy(), tau_grid=tau.copy(), values=values)


def _crank_nicolson_step(
    V_prev: np.ndarray,
    *,
    dt: float,
    sigma: float,
    S: np.ndarray,
    dS: float,
    r: float,
    K: float,
    tau_next: float,
) -> np.ndarray:
    N = len(S)
    tau_curr = tau_next - dt
    V_prev_bc = V_prev.copy()
    V_prev_bc[0] = 0.0
    V_prev_bc[-1] = S[-1] - K * np.exp(-r * tau_curr)

    m = N - 2
    rhs = np.zeros(m)
    lower = np.zeros(m - 1)
    diag = np.zeros(m)
    upper = np.zeros(m - 1)

    sigma2 = sigma ** 2
    V_right_next = S[-1] - K * np.exp(-r * tau_next)

    for idx in range(1, N - 1):
        S_i = S[idx]
        i = idx - 1
        coeff = sigma2 * (S_i ** 2)
        alpha = 0.25 * dt * (coeff / (dS ** 2) - r * S_i / dS)
        beta = -0.5 * dt * (coeff / (dS ** 2) + r)
        gamma = 0.25 * dt * (coeff / (dS ** 2) + r * S_i / dS)

        diag[i] = 1.0 - beta
        if idx > 1:
            lower[i - 1] = -alpha
        if idx < N - 2:
            upper[i] = -gamma

        rhs_i = (
            alpha * V_prev_bc[idx - 1]
            + (1.0 + beta) * V_prev_bc[idx]
            + gamma * V_prev_bc[idx + 1]
        )
        if idx == N - 2:
            rhs_i += gamma * V_right_next
        rhs[i] = rhs_i

    solution = _solve_tridiagonal(lower, diag, upper, rhs)

    V_next = np.zeros_like(V_prev)
    V_next[0] = 0.0
    V_next[-1] = V_right_next
    V_next[1:-1] = solution
    return V_next


def _solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    n = len(diag)
    d = diag.astype(float).copy()
    a = lower.astype(float).copy()
    c = upper.astype(float).copy()
    b = rhs.astype(float).copy()

    for i in range(1, n):
        w = a[i - 1] / d[i - 1]
        d[i] -= w * c[i - 1]
        b[i] -= w * b[i - 1]

    x = np.zeros(n, dtype=float)
    x[-1] = b[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        coef = c[i] if i < len(c) else 0.0
        x[i] = (b[i] - coef * x[i + 1]) / d[i]
    return x


def _round_tau(tau: float) -> float:
    return float(np.round(float(tau), 10))
