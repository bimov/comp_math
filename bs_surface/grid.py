# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from .dupire import solve_dupire_crank_nicolson, _round_tau
from .pricing import call_black_scholes
from .calibration import calibrate_local_volatility


def build_grids(df: pd.DataFrame, n_s: int, n_t: int, maturity: str):
    s_lo = df["Close"].quantile(0.05)
    s_hi = df["Close"].quantile(0.95)

    if isinstance(s_lo, (pd.Series, pd.DataFrame)):
        s_lo = float(s_lo.iloc[0])
    if isinstance(s_hi, (pd.Series, pd.DataFrame)):
        s_hi = float(s_hi.iloc[0])

    S_grid = np.linspace(float(s_lo), float(s_hi), n_s)

    all_dates = df.index
    T_date = pd.to_datetime(maturity)
    all_dates = all_dates[all_dates < T_date]
    if len(all_dates) == 0:
        raise ValueError("Все t_j >= T. Выберите maturity позже, чем даты котировок.")
    idx = np.linspace(0, len(all_dates) - 1, n_t).round().astype(int)
    t_grid = all_dates[idx]
    sigma_series = df["sigma_annual"]
    return S_grid, t_grid, T_date, sigma_series


def build_surface(S_grid, t_grid, T_date, r, K, sigma_series) -> pd.DataFrame:
    tau_nodes, sigma_surface, sigma_curve = _build_tau_sigma_surface(
        S_grid, t_grid, T_date, sigma_series
    )
    w_market = sigma_curve ** 2
    C_market_surface = _build_market_surface(S_grid, tau_nodes, sigma_curve, r=r, K=K)

    calib_result = calibrate_local_volatility(S_grid, tau_nodes=tau_nodes, w_market=w_market, r=r, K=K, initial_sigma=sigma_curve)
    dupire_solution = solve_dupire_crank_nicolson(S_grid, tau_nodes=tau_nodes, sigma_surface=np.repeat(calib_result.sigma_curve[:, None], len(S_grid), axis=1), r=r, K=K)

    market_map = {_round_tau(t): row for t, row in zip(tau_nodes, C_market_surface)}
    model_map = {_round_tau(t): dupire_solution.get_values(t) for t in tau_nodes}
    sigma_map = {_round_tau(t): s for t, s in zip(tau_nodes, calib_result.sigma_curve)}

    records = []
    for t in t_grid:
        tau = _time_to_maturity(T_date, t)
        tau_key = _round_tau(tau)

        C_market_vals = market_map[tau_key]
        C_model_vals = model_map[tau_key]
        sigma_t = float(sigma_map[tau_key])

        for S, C_mkt, C_cn in zip(S_grid, C_market_vals, C_model_vals):
            records.append(
                {
                    "date": t.date(),
                    "tau_yrs": tau,
                    "S": float(S),
                    "sigma": sigma_t,
                    "C_market": float(C_mkt),
                    "C_model": float(C_cn),
                }
            )

    df = pd.DataFrame.from_records(records)
    for col in ("S", "C_market", "C_model", "tau_yrs", "sigma"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().sort_values(["date", "S"])
    df.attrs["calibration_rms_history"] = calib_result.rms_history
    df.attrs["calibration_success"] = calib_result.success
    df.attrs["calibration_message"] = calib_result.message
    df.attrs["w_market"] = w_market
    df.attrs["w_model"] = calib_result.w_model
    df.attrs["v_curve"] = calib_result.v_curve
    return df


def _build_tau_sigma_surface(S_grid, t_grid, T_date, sigma_series):
    """
    Построить узлы по tau и поверхность локальной волатильности sigma(S, tau).
    Возвращает также вектор sigma(tau) для калибровки.
    """
    sigma_series = sigma_series.sort_index()
    if sigma_series.empty:
        raise ValueError("Серия волатильности пуста")

    sigma_at_maturity = float(sigma_series.iloc[-1])

    pairs = [(0.0, max(sigma_at_maturity, 1e-8))]
    for t in sorted(t_grid):
        tau = _time_to_maturity(T_date, t)
        if tau <= 0:
            continue

        hist = sigma_series.loc[:t]
        if hist.empty:
            sigma_t = sigma_at_maturity
        else:
            sigma_t = float(hist.iloc[-1])

        sigma_t = max(sigma_t, 1e-8)
        pairs.append((tau, sigma_t))

    pairs = _deduplicate_tau_sigma(pairs)
    tau_nodes = np.array([tau for tau, _ in pairs], dtype=float)
    sigma_values = np.array([sigma for _, sigma in pairs], dtype=float)

    N = len(S_grid)
    sigma_surface = np.repeat(sigma_values[:, None], N, axis=1)

    return tau_nodes, sigma_surface, sigma_values


def _time_to_maturity(T_date, t) -> float:
    return max((T_date - t).days / 365.25, 0.0)


def _deduplicate_tau_sigma(pairs, tol=1e-10):
    pairs = sorted(pairs, key=lambda x: x[0])
    dedup = []
    for tau, sigma in pairs:
        if dedup and abs(tau - dedup[-1][0]) <= tol:
            dedup[-1] = (tau, sigma)
        else:
            dedup.append((tau, sigma))
    return dedup


def _build_market_surface(S_grid, tau_nodes, sigma_curve, r, K):
    """
    Сгенерировать «рыночную» поверхность C_market из BS цен.
    """

    C_market = []
    for tau, sigma in zip(tau_nodes, sigma_curve):
        sigma = max(float(sigma), 1e-8)
        C_vals = call_black_scholes(S_grid, K=K, r=r, sigma=sigma, tau=float(tau))
        C_market.append(C_vals)
    return np.asarray(C_market, dtype=float)
