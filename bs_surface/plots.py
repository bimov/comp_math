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
    ax.set_ylabel("time index")
    ax.set_zlabel(f"{value_col}(S,t)")
    ax.set_title(_surface_title(value_col, suffix="(S,t)"))
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
    plt.title(_surface_title(value_col, suffix="(heatmap)"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def plot_rms(surface: pd.DataFrame, out_path: str) -> None:
    """Построить график RMS ошибки между C_model и C_market по датам."""

    if not {"C_model", "C_market"}.issubset(surface.columns):
        raise KeyError("Surface must contain C_model и C_market для RMS графика")

    rms_by_date = surface.groupby("date").apply(
        lambda df: float(np.sqrt(np.mean((df["C_model"] - df["C_market"]) ** 2)))
    )

    plt.figure(figsize=(8, 4))
    rms_by_date.plot(marker="o")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlabel("Дата")
    plt.ylabel("RMS(C_model - C_market)")
    plt.title("Качество калибровки волатильности")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_model_vs_market(surface: pd.DataFrame, out_path: str) -> None:
    required_cols = {"date", "C_model", "C_market"}
    if not required_cols.issubset(surface.columns):
        raise KeyError("Surface must contain date, C_model и C_market для сравнения графиков")

    by_date = surface.groupby("date")[["C_model", "C_market"]].mean()

    plt.figure(figsize=(10, 4))
    by_date["C_model"].plot(marker="o", label="C_model")
    by_date["C_market"].plot(marker="s", label="C_market")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlabel("Дата")
    plt.ylabel("Средняя цена опциона")
    plt.title("Сравнение C_model и C_market")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def _choose_value_column(surface: pd.DataFrame) -> str:
    for col in ("C_model", "C_bs", "C"):
        if col in surface.columns:
            return col
    raise KeyError("Не найдено колонок с ценой опциона")


def _surface_title(value_col: str, suffix: str) -> str:
    name_map = {
        "C_model": "Crank–Nicolson surface",
        "C_bs": "Black–Scholes surface",
        "C": "Option surface",
    }
    prefix = name_map.get(value_col, "Option surface")
    return f"{prefix} {value_col} {suffix}"
