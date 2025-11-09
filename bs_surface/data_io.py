# -*- coding: utf-8 -*-
import os
from typing import Tuple
import numpy as np
import pandas as pd
import yfinance as yf


def ensure_out_dir(base: str) -> str:
    out_dir = os.path.join(base, "out")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def download_quotes(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Скачать дневные котировки и оставить только скорректированную цену закрытия.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise RuntimeError("Пустые котировки. Проверьте тикер/интернет.")
    df = df[["Close"]].copy()
    df.index.name = "Date"
    df["Close"] = df["Close"].astype(float)
    return df.dropna()


def add_returns_and_vol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавить лог-доходности и годовую волатильность к ряду цен.

    Волатильность оценивается как стандартное отклонение лог-доходностей
    на 21-дневном скользящем окне, умноженное на sqrt(252) для годовой шкалы.
    """
    df = df.copy()
    df["log_ret"] = np.log(df["Close"]).diff()
    df["sigma_annual"] = df["log_ret"].rolling(21).std(ddof=0) * np.sqrt(252.0)
    df["sigma_annual"] = df["sigma_annual"].bfill()
    return df


def choose_strike(df: pd.DataFrame, policy: str, strike_fixed: float | None) -> float:
    if policy == "median":
        return df["Close"].median(skipna=True).item()
    if policy == "atm_first":
        return float(df["Close"].iloc[0])
    if policy == "fixed":
        return float(strike_fixed)
    raise ValueError("Unknown strike policy")
