# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def plot_price_series(df: pd.DataFrame, out_path: str) -> None:
    """
    Нарисовать график временного ряда цены и сохранить в файл. (График цены акции)
    """
    plt.figure(figsize=(10, 4))
    df["Close"].plot()
    plt.title("AAPL Close (adjusted)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _pivot_surface(surface: pd.DataFrame):
    """
    Функция строит сводную таблицу, чтобы получить матрицу значений C(S,t).
    """
    value_col = _choose_value_column(surface)
    pivot = surface.pivot(index="date", columns="S", values=value_col)
    pivot = pivot.sort_index().sort_index(axis=1)
    Z = pivot.to_numpy()
    S_vals = pivot.columns.to_numpy(dtype=float)
    dates = list(pivot.index)
    return S_vals, dates, Z, value_col


def plot_surface_3d(surface: pd.DataFrame, out_path: str) -> None:
    """
    Построить и сохранить 3D-поверхность C(S,t).
    """
    S_vals, dates, Z, value_col = _pivot_surface(surface)
    X, Y = np.meshgrid(S_vals, np.arange(len(dates)))
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
    ax.set_xlabel("S (price)")
    ax.set_zlabel(f"{value_col}(S,t)")
    ax.set_title(f"Option surface {value_col}(S,t)")
    ax.set_title("Black–Scholes surface C(S,t)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def plot_heatmap(surface: pd.DataFrame, out_path: str) -> None:
    """
    Построить «тепловую карту» C(S,t) и сохранить в файл.
    Цвет соответствует уровню цены опциона C. Оси подписаны значениями S и датами.
    """
    S_vals, dates, Z, value_col = _pivot_surface(surface)
    plt.figure(figsize=(10, 6))
    plt.imshow(Z, aspect="auto", origin="lower")
    plt.colorbar(label=f"{value_col}(S,t)")
    yticks = np.linspace(0, len(dates) - 1, num=min(10, len(dates))).astype(int)
    xticks = np.linspace(0, len(S_vals) - 1, num=min(10, len(S_vals))).astype(int)
    plt.yticks(ticks=yticks, labels=[str(dates[i]) for i in yticks])
    plt.xticks(
        ticks=xticks,
        labels=[f"{S_vals[i]:.2f}" for i in xticks],
        rotation=45,
        ha="right",
    )
    plt.xlabel("S grid")
    plt.ylabel("t (dates)")
    plt.title(f"Option surface {value_col} (heatmap)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def _choose_value_column(surface: pd.DataFrame) -> str:
    for col in ("C_model", "C_bs", "C"):
        if col in surface.columns:
            return col
    raise KeyError("Не найдено колонок с ценой опциона")
