"""One-off training script: reproduces the notebook's models and saves artifacts for the app.

Usage (from C:\\BKU\\DM):
    app\\.venv\\Scripts\\python app\\train.py            # full run (bootstrap + no-age experiment)
    app\\.venv\\Scripts\\python app\\train.py --fast     # skip the slow extras
    app\\.venv\\Scripts\\python app\\train.py --data path\\to\\Base.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fraud_core import drift  # noqa: E402
from fraud_core.metrics import MAX_FPR, evaluate, threshold_at_fpr, tpr_at_fpr  # noqa: E402
from fraud_core.preprocessing import (BASE_NUM, BIN_COLS, CAT_COLS, FEATURE_COLS,  # noqa: E402
                                      LR_NUM_COLS, MASKED_COLS, TARGET, clean, make_X, to_lgbm)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
SEEDS = [42, 202, 777]
LGBM_PARAMS = dict(n_estimators=3000, learning_rate=0.02, num_leaves=63, min_child_samples=100,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
                   n_jobs=-1, verbose=-1)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_data(arg: str | None) -> Path:
    candidates = [Path(arg)] if arg else []
    candidates += [ROOT.parent / "data" / "Base.csv", ROOT.parent / "tabular" / "data" / "Base.csv"]
    for c in candidates:
        if c.exists():
            return c
    import kagglehub
    log("Base.csv not found locally - downloading via kagglehub ...")
    return Path(kagglehub.dataset_download("sgpjesus/bank-account-fraud-dataset-neurips-2022")) / "Base.csv"


def build_lr() -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric, LR_NUM_COLS), ("cat", categorical, CAT_COLS)])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(class_weight="balanced", C=1.0,
                                                                max_iter=5000, tol=1e-3, random_state=42))])


def train_lgbm(X_tr, y_tr, X_va, y_va, cols):
    models = []
    for seed in SEEDS:
        m = lgb.LGBMClassifier(random_state=seed, **LGBM_PARAMS)
        m.fit(X_tr[cols], y_tr, eval_set=[(X_va[cols], y_va)], eval_metric="auc",
              categorical_feature=[c for c in CAT_COLS if c in cols],
              callbacks=[lgb.early_stopping(200, verbose=False)])
        models.append(m)
    return models


def avg_predict(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def fairness_rows(y, pred, ages, policy):
    rows = []
    for g, mask in [("< 50", ages < 50), (">= 50", ages >= 50), ("Tổng", np.ones_like(ages, bool))]:
        yt, yp = y[mask], pred[mask]
        rows.append({"policy": policy, "group": g, "N": int(mask.sum()),
                     "TPR": float(np.sum(yp & (yt == 1)) / max(yt.sum(), 1)),
                     "FPR": float(np.sum(yp & (yt == 0)) / max((yt == 0).sum(), 1))})
    return rows


def bootstrap(y, s_lr, s_lgb, thr_lr, thr_lgb, n_boot):
    rng = np.random.default_rng(42)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]

    def bundle(yb, s, thr):
        fpr, tpr, _ = roc_curve(yb, s)
        j = np.where(fpr <= MAX_FPR)[0][-1]
        return tpr[j], (s >= thr)[yb == 1].mean(), auc(fpr, tpr)

    lr, lg = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
        yb = y[idx]
        lr.append(bundle(yb, s_lr[idx], thr_lr))
        lg.append(bundle(yb, s_lgb[idx], thr_lgb))
    lr, lg = np.array(lr), np.array(lg)
    out = {}
    for k, name in enumerate(["TPR@FPR5% (xếp hạng)", "TPR@ngưỡng", "ROC-AUC"]):
        d = lg[:, k] - lr[:, k]
        out[name] = {"lr_ci": np.percentile(lr[:, k], [2.5, 97.5]).tolist(),
                     "lgbm_ci": np.percentile(lg[:, k], [2.5, 97.5]).tolist(),
                     "diff_ci": np.percentile(d, [2.5, 97.5]).tolist()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--fast", action="store_true", help="skip bootstrap and no-age experiment")
    args = ap.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)

    path = find_data(args.data)
    log(f"Loading {path}")
    raw = pd.read_csv(path)
    df = clean(raw)
    tr, va, te = df[df.month <= 5], df[df.month == 6], df[df.month == 7]
    log(f"Train {len(tr):,} | Val {len(va):,} | Test {len(te):,}")

    categories = {c: sorted(tr[c].astype(str).unique().tolist()) for c in CAT_COLS}
    X_tr, X_va, X_te = make_X(tr), make_X(va), make_X(te)
    y_tr, y_va, y_te = tr[TARGET].to_numpy(), va[TARGET].to_numpy(), te[TARGET].to_numpy()
    L_tr, L_va, L_te = (to_lgbm(x, categories) for x in (X_tr, X_va, X_te))

    log("Training Logistic Regression baseline ...")
    lr = build_lr().fit(X_tr, y_tr)
    lr_va, lr_te = lr.predict_proba(X_va)[:, 1], lr.predict_proba(X_te)[:, 1]
    lr_thr = threshold_at_fpr(y_va, lr_va)
    joblib.dump(lr, ARTIFACTS / "lr_pipe.joblib")

    log("Training LightGBM x3 seeds ...")
    models = train_lgbm(L_tr, y_tr, L_va, y_va, FEATURE_COLS)
    best_iters = [int(m.best_iteration_) for m in models]
    lg_va, lg_te = avg_predict(models, L_va), avg_predict(models, L_te)
    lg_tr = avg_predict(models, L_tr)
    model_files = []
    for seed, m in zip(SEEDS, models):
        f = f"lgbm_seed{seed}.txt"
        m.booster_.save_model(str(ARTIFACTS / f), num_iteration=m.best_iteration_)
        model_files.append(f)

    age_va, age_te = va.customer_age.to_numpy(), te.customer_age.to_numpy()
    thr = {"common": threshold_at_fpr(y_va, lg_va),
           "young": threshold_at_fpr(y_va[age_va < 50], lg_va[age_va < 50]),
           "old": threshold_at_fpr(y_va[age_va >= 50], lg_va[age_va >= 50])}
    log(f"Best iterations {best_iters} | threshold(common) {thr['common']:.4f}")

    results = {"val": {"Logistic Regression": evaluate(y_va, lr_va, lr_thr),
                       "LightGBM": evaluate(y_va, lg_va, thr["common"])},
               "test": {"Logistic Regression": evaluate(y_te, lr_te, lr_thr),
                        "LightGBM": evaluate(y_te, lg_te, thr["common"])}}
    t = results["test"]["LightGBM"]
    log(f"Test LightGBM: TPR@5%FPR {t['TPR@FPR5% (xếp hạng)']:.4f} | TPR@thr {t['TPR@ngưỡng']:.4f} "
        f"| ROC-AUC {t['ROC-AUC']:.4f}")

    pred_common = lg_te >= thr["common"]
    pred_group = np.where(age_te < 50, lg_te >= thr["young"], lg_te >= thr["old"])
    fairness = fairness_rows(y_te, pred_common, age_te, "common") + \
        fairness_rows(y_te, pred_group, age_te, "group")

    extras = {}
    if not args.fast:
        log("Bootstrap 1,000 resamples on Test ...")
        extras["bootstrap"] = bootstrap(y_te, lr_te, lg_te, lr_thr, thr["common"], 1000)
        log("No-age experiment + proxy leakage ...")
        no_age = [c for c in FEATURE_COLS if c not in ("customer_age", "addr_per_age", "bank_per_age")]
        na_models = train_lgbm(L_tr, y_tr, L_va, y_va, no_age)
        na_va, na_te = avg_predict(na_models, L_va[no_age]), avg_predict(na_models, L_te[no_age])
        na_thr = threshold_at_fpr(y_va, na_va)
        rows = fairness_rows(y_te, na_te >= na_thr, age_te, "no_age")
        proxy = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                   random_state=42, verbose=-1)
        proxy.fit(L_tr[no_age], (tr.customer_age >= 50).astype(int),
                  categorical_feature=CAT_COLS)
        extras["no_age"] = {"fairness": rows,
                            "tpr_at_fpr": tpr_at_fpr(y_te, na_te),
                            "proxy_auc": float(roc_auc_score(age_te >= 50,
                                                             proxy.predict_proba(L_te[no_age])[:, 1]))}

    imp = np.mean([m.booster_.feature_importance(importance_type="gain") for m in models], axis=0)
    importance = sorted(zip(FEATURE_COLS, (imp / imp.sum()).tolist()), key=lambda x: -x[1])

    # Form defaults and widget bounds (raw scale; -1 / negative = missing)
    raw_tr = raw[raw.month <= 5]
    form = {}
    for c in BASE_NUM + BIN_COLS:
        s = tr[c].dropna()
        form[c] = {"median": float(s.median()), "min": float(raw_tr[c].min()), "max": float(raw_tr[c].max()),
                   "missing_rate": float(tr[c].isna().mean()) if c in MASKED_COLS else 0.0,
                   "integer": bool(pd.api.types.is_integer_dtype(raw_tr[c]))}
    for c in CAT_COLS:
        form[c] = {"mode": str(tr[c].mode()[0]), "levels": categories[c]}

    reference = {c: drift.numeric_reference(tr[c]) for c in BASE_NUM}
    reference.update({c: drift.categorical_reference(tr[c]) for c in BIN_COLS + CAT_COLS})
    score_ref = drift.numeric_reference(pd.Series(lg_tr))

    monthly = (df.groupby("month")[TARGET].agg(N="count", fraud="sum", rate="mean")
               .reset_index().to_dict(orient="records"))

    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "data_path": str(path),
        "model_files": model_files,
        "best_iterations": best_iters,
        "categories": categories,
        "thresholds": thr,
        "lr_threshold": lr_thr,
        "results": results,
        "brier": {"LightGBM": float(brier_score_loss(y_te, lg_te)),
                  "Logistic Regression": float(brier_score_loss(y_te, lr_te))},
        "fairness": fairness,
        "importance": importance,
        "form": form,
        "reference": reference,
        "score_reference": score_ref,
        "monthly": monthly,
        "train_fraud_rate": float(y_tr.mean()),
        **extras,
    }
    (ARTIFACTS / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float),
                                         encoding="utf-8")

    pd.DataFrame({"split": ["val"] * len(va) + ["test"] * len(te),
                  "y": np.concatenate([y_va, y_te]),
                  "lgbm": np.concatenate([lg_va, lg_te]),
                  "lr": np.concatenate([lr_va, lr_te]),
                  "customer_age": np.concatenate([age_va, age_te]),
                  "proposed_credit_limit": np.concatenate([va.proposed_credit_limit, te.proposed_credit_limit]),
                  }).to_parquet(ARTIFACTS / "scores.parquet", index=False)

    raw[raw.month >= 6].reset_index(drop=True).to_parquet(ARTIFACTS / "samples_raw.parquet", index=False)
    raw[raw.month == 7].sample(10_000, random_state=7).to_csv(ARTIFACTS / "sample_batch_month7.csv",
                                                             index=False)
    log(f"Saved artifacts to {ARTIFACTS}")


if __name__ == "__main__":
    main()
