#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
import os
from bs_surface.data_io import ensure_out_dir, download_quotes, add_returns_and_vol, choose_strike
from bs_surface.grid import build_grids, build_surface
from bs_surface.plots import plot_price_series, plot_surface_3d, plot_heatmap, plot_rms


@dataclass
class Config:
    ticker: str = "AAPL"
    start: str = "2017-01-01"
    end: str = "2019-12-31"
    maturity: str = "2020-01-31"
    r: float = 0.02
    n_s: int = 60
    n_t: int = 40
    strike_policy: str = "median"
    strike_fixed: float | None = None

def parse_args() -> Config:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ticker", default=Config.ticker)
    p.add_argument("--start", default=Config.start)
    p.add_argument("--end", default=Config.end)
    p.add_argument("--maturity", default=Config.maturity)
    p.add_argument("--r", type=float, default=Config.r)
    p.add_argument("--n-s", type=int, default=Config.n_s, dest="n_s")
    p.add_argument("--n-t", type=int, default=Config.n_t, dest="n_t")
    p.add_argument("--strike-policy", choices=["median", "atm_first", "fixed"], default=Config.strike_policy)
    p.add_argument("--strike-fixed", type=float, default=None)
    args = p.parse_args()
    if args.strike_policy == "fixed" and args.strike_fixed is None:
        p.error("Для --strike-policy=fixed укажите --strike-fixed=<число>")
    return Config(**vars(args))

def main(cfg: Config):
    """
    Функция запускает весь пайплайн.
    """
    out_dir = ensure_out_dir(os.path.dirname(__file__))
    df = download_quotes(cfg.ticker, cfg.start, cfg.end)
    df = add_returns_and_vol(df)
    df.to_csv(os.path.join(out_dir, "quotes.csv"), index=True)

    S_grid, t_grid, T_date, sigma_series = build_grids(df, cfg.n_s, cfg.n_t, cfg.maturity)
    K = choose_strike(df, cfg.strike_policy, cfg.strike_fixed)
    surface = build_surface(S_grid, t_grid, T_date, r=cfg.r, K=K, sigma_series=sigma_series)
    surface.to_csv(os.path.join(out_dir, "bs_surface.csv"), index=False)

    plot_price_series(df, os.path.join(out_dir, "price_series.png"))
    plot_surface_3d(surface, os.path.join(out_dir, "bs_surface_3d.png"))
    plot_heatmap(surface, os.path.join(out_dir, "bs_surface_heatmap.png"))

    print(f"Готово. Файлы сохранены в: {out_dir}")
    print(f"Страйк K = {K:.4f}, r = {cfg.r:.4f}")
    print(f"S_grid: [{S_grid[0]:.2f} .. {S_grid[-1]:.2f}] (n={len(S_grid)})")
    print(f"t_grid: {t_grid[0].date()} → {t_grid[-1].date()} (n={len(t_grid)}), T = {T_date.date()}")

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
