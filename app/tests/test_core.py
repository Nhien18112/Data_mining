import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from fraud_core import drift  # noqa: E402
from fraud_core.metrics import threshold_at_fpr  # noqa: E402
from fraud_core.preprocessing import (FEATURE_COLS, MASKED_COLS, RAW_INPUT_COLS, clean,  # noqa: E402
                                      make_X, to_lgbm, validate_raw)

CATS = {"payment_type": ["AA", "AB", "AC", "AD", "AE"], "employment_status": ["CA", "CB", "CC"],
        "housing_status": ["BA", "BB", "BC"], "source": ["INTERNET", "TELEAPP"],
        "device_os": ["linux", "macintosh", "other", "windows", "x11"]}


def synthetic(n=4):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.integers(1, 50, n).astype(float) for c in RAW_INPUT_COLS})
    for c in ["email_is_free", "phone_home_valid", "phone_mobile_valid", "has_other_cards",
              "foreign_request", "keep_alive_session"]:
        df[c] = rng.integers(0, 2, n)
    df["income"] = 0.5
    df["customer_age"] = 40
    for c in CATS:
        df[c] = CATS[c][0]
    df["device_fraud_count"] = 0
    df.loc[0, "prev_address_months_count"] = -1
    df.loc[1, "intended_balcon_amount"] = -7.3
    return df


def test_features_match_notebook_shape():
    X = make_X(clean(synthetic()))
    assert list(X.columns) == FEATURE_COLS
    assert X.shape[1] == 42


def test_missing_codes_become_nan_and_flags():
    c = clean(synthetic())
    assert np.isnan(c.loc[0, "prev_address_months_count"]) and c.loc[0, "prev_address_months_count_flag"] == 1
    assert np.isnan(c.loc[1, "intended_balcon_amount"]) and c.loc[1, "intended_balcon_amount_flag"] == 1
    assert c.loc[2, [f"{m}_flag" for m in MASKED_COLS]].sum() == 0
    assert "device_fraud_count" not in c.columns
    assert (c["month"] == 5).all()  # default month for new applications


def test_unknown_category_becomes_missing():
    raw = synthetic()
    raw.loc[0, "device_os"] = "beos"
    X = to_lgbm(make_X(clean(raw)), CATS)
    assert pd.isna(X.loc[0, "device_os"])
    _, warnings = validate_raw(raw, CATS)
    assert any("device_os" in w for w in warnings)


def test_validate_rejects_missing_column():
    errors, _ = validate_raw(synthetic().drop(columns=["income"]), CATS)
    assert errors and "income" in errors[0]


def test_threshold_respects_fpr_budget():
    rng = np.random.default_rng(1)
    y = (rng.random(20_000) < 0.02).astype(int)
    s = rng.random(20_000) + y * 0.5
    thr = threshold_at_fpr(y, s, 0.05)
    assert ((s >= thr) & (y == 0)).sum() / (y == 0).sum() <= 0.05


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(2)
    a = pd.Series(rng.normal(size=50_000))
    ref = drift.numeric_reference(a)
    same = drift.psi(ref["expected"], drift._numeric_shares(a, ref["edges"]))
    shifted = drift.psi(ref["expected"], drift._numeric_shares(a + 1.5, ref["edges"]))
    assert same < 1e-9 and shifted > 0.25


ARTIFACTS_READY = (APP / "artifacts" / "meta.json").exists()


@pytest.mark.skipif(not ARTIFACTS_READY, reason="run app/train.py first")
def test_model_scores_and_explanations_add_up():
    from fraud_core.explain import contributions
    from fraud_core.model import FraudModel

    model = FraudModel.load()
    raw = pd.read_csv(APP / "artifacts" / "sample_batch_month7.csv").head(50)
    scores, _, X = model.score(raw)
    assert scores.shape == (50,) and ((scores >= 0) & (scores <= 1)).all()
    contrib, bias = contributions(model.boosters, X)
    raw_margin = np.mean([b.predict(X, raw_score=True) for b in model.boosters], axis=0)
    np.testing.assert_allclose(contrib.sum(axis=1) + bias, raw_margin, atol=1e-6)


@pytest.mark.skipif(not ARTIFACTS_READY, reason="run app/train.py first")
def test_saved_scores_reproduce_thresholds():
    from fraud_core.model import FraudModel

    model = FraudModel.load()
    scores = pd.read_parquet(APP / "artifacts" / "scores.parquet")
    val = scores[scores.split == "val"]
    assert threshold_at_fpr(val.y, val.lgbm) == pytest.approx(model.meta["thresholds"]["common"])
    # Re-scoring the stored raw test rows through the app path must reproduce training-time scores
    raw = pd.read_parquet(APP / "artifacts" / "samples_raw.parquet")
    rescored, _, _ = model.score(raw[raw.month == 7].head(2000))
    np.testing.assert_allclose(rescored, scores[scores.split == "test"].lgbm.to_numpy()[:2000], atol=1e-9)
