from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable
import numpy as np


@dataclass
class DupireSolution:
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


def solve_dupire_crank_nicolson(
    S_grid: Iterable[float],
    tau_nodes: Iterable[float],
    sigma_surface: np.ndarray,
    r: float,
    K: float,
) -> DupireSolution:
    """
    Решить уравнение Дюпира на сетке (S, tau) методом Крэнка—Николсона
    для локальной волатильности sigma(S, tau).

    sigma_surface: массив формы (len(tau_nodes), len(S_grid)),
    где sigma_surface[n, i] = sigma(S_i, tau_n).
    """
    S = np.asarray(S_grid, dtype=float)
    if S.ndim != 1 or len(S) < 3:
        raise ValueError("Сетка S должна содержать минимум 3 узла")

    tau = np.asarray(list(tau_nodes), dtype=float)

    sigma_surface = np.asarray(sigma_surface, dtype=float)
    if sigma_surface.shape != (len(tau), len(S)):
        raise ValueError(
            f"sigma_surface должно иметь форму (len(tau_nodes), len(S_grid)) = ({len(tau)}, {len(S)}), "
            f"а сейчас {sigma_surface.shape}"
        )

    if tau[0] != 0.0:
        raise ValueError("Первая точка по времени должна быть tau=0")
    if not np.all(np.diff(tau) >= 0):
        raise ValueError("Сетка tau должна быть неубывающей")

    dS = float(S[1] - S[0])
    if not np.allclose(np.diff(S), dS, atol=1e-12):
        raise ValueError("Сетка S предполагается равномерной")

    sigma_surface = np.maximum(sigma_surface, 1e-8)

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

        sigma_prev = sigma_surface[n]      # shape (N,)
        sigma_next = sigma_surface[n + 1]  # shape (N,)

        V_next = _crank_nicolson_step_localvol(
            V_prev,
            dt=dt,
            sigma_prev=sigma_prev,
            sigma_next=sigma_next,
            S=S,
            dS=dS,
            r=r,
            K=K,
            tau_next=tau_next,
        )
        values[_round_tau(tau_next)] = V_next.copy()
        V_prev = V_next

    return DupireSolution(S_grid=S.copy(), tau_grid=tau.copy(), values=values)


def _crank_nicolson_step_localvol(
    V_prev: np.ndarray,
    *,
    dt: float,
    sigma_prev: np.ndarray,
    sigma_next: np.ndarray,
    S: np.ndarray,
    dS: float,
    r: float,
    K: float,
    tau_next: float,
) -> np.ndarray:
    """
    Один шаг Крэнка–Николсона для локальной волатильности sigma(S, tau).

    (I - 0.5 dt L_next) V_next = (I + 0.5 dt L_prev) V_prev
    """
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

    sigma2_prev = sigma_prev ** 2
    sigma2_next = sigma_next ** 2

    V_right_next = S[-1] - K * np.exp(-r * tau_next)

    for idx in range(1, N - 1):
        S_i = S[idx]
        i = idx - 1

        # коэффициенты оператора на старом и новом слое
        coeff_prev = sigma2_prev[idx] * (S_i ** 2)
        coeff_next = sigma2_next[idx] * (S_i ** 2)

        alpha_prev = 0.25 * dt * (coeff_prev / (dS ** 2) - r * S_i / dS)
        beta_prev  = -0.5 * dt * (coeff_prev / (dS ** 2) + r)
        gamma_prev = 0.25 * dt * (coeff_prev / (dS ** 2) + r * S_i / dS)

        alpha_next = 0.25 * dt * (coeff_next / (dS ** 2) - r * S_i / dS)
        beta_next  = -0.5 * dt * (coeff_next / (dS ** 2) + r)
        gamma_next = 0.25 * dt * (coeff_next / (dS ** 2) + r * S_i / dS)

        # матрица A = (I - 0.5 dt L_next)
        diag[i] = 1.0 - beta_next
        if idx > 1:
            lower[i - 1] = -alpha_next
        if idx < N - 2:
            upper[i] = -gamma_next

        # правая часть: (I + 0.5 dt L_prev) V_prev
        rhs_i = (
            alpha_prev * V_prev_bc[idx - 1]
            + (1.0 + beta_prev) * V_prev_bc[idx]
            + gamma_prev * V_prev_bc[idx + 1]
        )

        # вклад правой границы на новом слое
        if idx == N - 2:
            rhs_i += gamma_next * V_right_next

        rhs[i] = rhs_i

    solution = _solve_tridiagonal(lower, diag, upper, rhs)

    V_next = np.zeros_like(V_prev)
    V_next[0] = 0.0
    V_next[-1] = V_right_next
    V_next[1:-1] = solution
    return V_next


def _solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Метод прогонки для трёхдиагональной системы."""
    n = len(diag)
    d = diag.astype(float).copy()
    a = lower.astype(float).copy()
    c = upper.astype(float).copy()
    b = rhs.astype(float).copy()

    # прямой ход
    for i in range(1, n):
        w = a[i - 1] / d[i - 1]
        d[i] -= w * c[i - 1]
        b[i] -= w * b[i - 1]

    # обратный ход
    x = np.zeros(n, dtype=float)
    x[-1] = b[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        coef = c[i] if i < len(c) else 0.0
        x[i] = (b[i] - coef * x[i + 1]) / d[i]
    return x


def _round_tau(tau: float) -> float:
    return float(np.round(float(tau), 10))
