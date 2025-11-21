from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from scipy.optimize import minimize

from .dupire import solve_dupire_crank_nicolson


@dataclass
class CalibrationResult:
    sigma_curve: np.ndarray
    v_curve: np.ndarray
    model_prices: np.ndarray
    w_model: np.ndarray
    rms_history: List[float]
    success: bool
    message: str


def calibrate_local_volatility(S_grid: Iterable[float], tau_nodes: Iterable[float], w_market: np.ndarray, *, r: float, K: float,
    initial_sigma: np.ndarray, maxiter: int = 60, smooth_reg: float = 1e-3, v_bounds: Tuple[float, float] = (1e-6, 25.0)) -> CalibrationResult:

    S_grid = np.asarray(S_grid, dtype=float)
    tau_nodes = np.asarray(list(tau_nodes), dtype=float)
    w_market = np.asarray(w_market, dtype=float)
    init_sigma = np.clip(np.asarray(initial_sigma, dtype=float), 1e-4, None)
    init_v = np.clip(init_sigma ** 2, v_bounds[0], v_bounds[1])

    if w_market.shape != (len(tau_nodes),):
        raise ValueError("длина w_market должна совпадать с (len(tau_nodes),)")

    bounds = [v_bounds] * len(init_v)
    history: List[float] = []

    def _sigma_surface(v_curve: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        v_clipped = np.clip(v_curve, v_bounds[0], v_bounds[1])
        sigma_curve = np.sqrt(v_clipped)
        sigma_surface = np.repeat(sigma_curve[:, None], len(S_grid), axis=1)
        return sigma_curve, sigma_surface

    def _integrated_variance(v_curve: np.ndarray) -> np.ndarray:
        v_clipped = np.clip(v_curve, v_bounds[0], v_bounds[1])
        cum_int = np.zeros_like(v_clipped)
        for i in range(1, len(v_clipped)):
            dt = float(tau_nodes[i] - tau_nodes[i - 1])
            if dt < 0:
                raise ValueError("tau_nodes должны быть неубывающими")
            cum_int[i] = cum_int[i - 1] + 0.5 * (v_clipped[i] + v_clipped[i - 1]) * dt
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(tau_nodes > 0, cum_int / tau_nodes, v_clipped[0])
        return w

    def objective(v_curve: np.ndarray) -> float:
        w_model = _integrated_variance(v_curve)
        diff = w_model - w_market
        rms = float(np.sqrt(np.mean(diff**2)))
        if smooth_reg > 0 and len(v_curve) >= 3:
            second_diff = np.diff(v_curve, n=2)
            rms += float(smooth_reg * np.sqrt(np.mean(second_diff**2)))
        history.append(rms)
        return rms

    opt_res = minimize(objective, x0=init_v, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-10})
    best_v = np.clip(opt_res.x, v_bounds[0], v_bounds[1])
    best_sigma, best_surface = _sigma_surface(best_v)
    best_solution = solve_dupire_crank_nicolson(S_grid, tau_nodes=tau_nodes, sigma_surface=best_surface, r=r, K=K)
    best_prices = np.vstack([best_solution.get_values(tau) for tau in tau_nodes])
    best_w_model = _integrated_variance(best_v)

    return CalibrationResult(sigma_curve=best_sigma, v_curve=best_v, model_prices=best_prices, w_model=best_w_model, rms_history=history, success=bool(opt_res.success), message=str(opt_res.message))
