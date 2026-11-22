"""Population Stability Index (PSI) against the training distribution (notebook cell 69 idea)."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4
PSI_STABLE, PSI_WATCH = 0.10, 0.25


def numeric_reference(series: pd.Series, n_bins: int = 10) -> dict:
    """Decile cut-points on non-missing train values + expected share per bin (last bin = missing)."""
    s = series.astype(float)
    vals = s.dropna()
    edges = np.unique(np.quantile(vals, np.linspace(0, 1, n_bins + 1))[1:-1]).tolist()
    return {"edges": edges, "expected": _numeric_shares(s, edges)}


def _numeric_shares(s: pd.Series, edges: list[float]) -> list[float]:
    s = s.astype(float)
    n = len(s)
    if n == 0:
        return [0.0] * (len(edges) + 2)
    idx = np.searchsorted(np.asarray(edges), s.dropna().to_numpy(), side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    return (np.append(counts, s.isna().sum()) / n).tolist()


def categorical_reference(series: pd.Series) -> dict:
    freq = series.astype(str).value_counts(normalize=True)
    return {"levels": freq.index.tolist(), "expected": freq.values.tolist()}


def _categorical_shares(s: pd.Series, levels: list[str]) -> list[float]:
    s = s.astype(str)
    shares = [float((s == lv).mean()) for lv in levels]
    shares.append(float((~s.isin(levels)).mean()))  # unseen levels
    return shares


def psi(expected, actual) -> float:
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_table(df: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """PSI of each referenced column of `df` vs train. `df` must already be cleaned (NaN for missing)."""
    rows = []
    for col, ref in reference.items():
        if col not in df.columns:
            continue
        if "edges" in ref:
            value = psi(ref["expected"], _numeric_shares(df[col], ref["edges"]))
        else:
            value = psi(ref["expected"] + [0.0], _categorical_shares(df[col], ref["levels"]))
        rows.append({"Thuộc tính": col, "PSI": value, "Mức": label(value)})
    return pd.DataFrame(rows).sort_values("PSI", ascending=False).reset_index(drop=True)


def shares_table(df: pd.DataFrame, col: str, ref: dict) -> pd.DataFrame:
    """Bin-by-bin expected (train) vs actual shares for one column."""
    if "edges" in ref:
        e = ref["edges"]
        names = [f"≤ {e[0]:.4g}"] + [f"({a:.4g}, {b:.4g}]" for a, b in zip(e[:-1], e[1:])] + [f"> {e[-1]:.4g}"]
        names.append("Thiếu")
        actual = _numeric_shares(df[col], e)
        expected = ref["expected"]
    else:
        names = ref["levels"] + ["(mới)"]
        actual = _categorical_shares(df[col], ref["levels"])
        expected = ref["expected"] + [0.0]
    return pd.DataFrame({"Khoảng": names, "Train": expected, "Lô mới": actual})


def label(value: float) -> str:
    if value < PSI_STABLE:
        return "Ổn định"
    if value < PSI_WATCH:
        return "Cần theo dõi"
    return "Trôi dạt"
