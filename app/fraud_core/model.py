"""Load trained artifacts and score raw applications."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .preprocessing import clean, make_X, to_lgbm

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


@dataclass
class FraudModel:
    boosters: list
    meta: dict

    @classmethod
    def load(cls, folder: Path = ARTIFACTS) -> "FraudModel":
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        boosters = [lgb.Booster(model_file=str(folder / f)) for f in meta["model_files"]]
        return cls(boosters, meta)

    @property
    def categories(self) -> dict:
        return self.meta["categories"]

    def features(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (cleaned raw frame, LightGBM-ready feature frame)."""
        cleaned = clean(raw)
        return cleaned, to_lgbm(make_X(cleaned), self.categories)

    def predict(self, X_lgb: pd.DataFrame) -> np.ndarray:
        return np.mean([b.predict(X_lgb) for b in self.boosters], axis=0)

    def score(self, raw: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
        cleaned, X = self.features(raw)
        return self.predict(X), cleaned, X


def default_policy(meta: dict) -> dict:
    t = meta["thresholds"]
    return {"mode": "common", "fpr_budget": 0.05,
            "thr_common": t["common"], "thr_young": t["young"], "thr_old": t["old"]}


def decide(scores: np.ndarray, ages: np.ndarray, policy: dict) -> np.ndarray:
    """Boolean flag per application under the active policy."""
    scores = np.asarray(scores)
    if policy["mode"] == "group":
        thr = np.where(np.asarray(ages) < 50, policy["thr_young"], policy["thr_old"])
        return scores >= thr
    return scores >= policy["thr_common"]


def threshold_for(age: float, policy: dict) -> float:
    if policy["mode"] == "group":
        return policy["thr_young"] if age < 50 else policy["thr_old"]
    return policy["thr_common"]


def risk_band(p: float, thr: float) -> str:
    if p >= max(2 * thr, 0.20):
        return "Rất cao"
    if p >= thr:
        return "Cao"
    if p >= thr / 2:
        return "Trung bình"
    return "Thấp"
