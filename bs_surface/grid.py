# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from .pricing import call_black_scholes


def build_grids(df: pd.DataFrame, n_s: int, n_t: int, maturity: str):
    """
    Построить сетки S_i и t_j из реальных данных.

    Из диапазона реальных цен берётся квантильный интервал [5%, 95%] и на нём
    равномерно размещается сетка S из n_s узлов. Сетка по времени t строится,
    выбирая равномерно n_t дат.
    """
    s_lo, s_hi = df["Close"].quantile(0.05), df["Close"].quantile(0.95)
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
    records = []
    for t in t_grid:
        tau = max((T_date - t).days / 365.25, 0.0)
        sigma_t = float(sigma_series.loc[:t].iloc[-1])
        sigma_t = max(sigma_t, 1e-8)
        C_vals = call_black_scholes(S_grid, K=K, r=r, sigma=sigma_t, tau=tau)
        for S, C in zip(S_grid, C_vals):
            records.append({"date": t.date(), "tau_yrs": tau, "S": float(S), "sigma": sigma_t, "C": float(C)})
    df = pd.DataFrame.from_records(records)
    for col in ("S", "C", "tau_yrs", "sigma"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().sort_values(["date", "S"])
