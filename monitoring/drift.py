from __future__ import annotations

import numpy as np
import pandas as pd
from alibi_detect.cd import ClassifierDrift, KSDrift
from sklearn.ensemble import RandomForestClassifier


def compute_ks_drift(
    X_ref: pd.DataFrame,
    X_prod: pd.DataFrame,
    features: list[str],
    p_val: float = 0.05,
) -> dict:
    ref = X_ref[features].astype(np.float32).values
    prod = X_prod[features].astype(np.float32).values

    detector = KSDrift(ref, p_val=p_val)
    res = detector.predict(prod)

    distances = res["data"]["distance"]
    p_values = res["data"]["p_val"]

    per_feature = []
    n_drifted = 0
    for feat, d, p in zip(features, distances, p_values):
        is_feat_drift = bool(p < p_val)
        n_drifted += int(is_feat_drift)
        per_feature.append({
            "feature": feat,
            "D_KS": float(d),
            "p_value": float(p),
            "drift": is_feat_drift,
        })

    return {
        "is_drift": bool(res["data"]["is_drift"]),
        "per_feature": per_feature,
        "n_drifted": n_drifted,
        "min_p_value": float(min(p_values)) if len(p_values) else 1.0,
    }


def compute_classifier_drift(
    X_ref: pd.DataFrame,
    X_prod: pd.DataFrame,
    features: list[str],
    p_val: float = 0.05,
    n_per_class: int = 1000,
) -> dict:
    ref = X_ref[features].astype(np.float32).values
    prod = X_prod[features].astype(np.float32).values

    # Submuestreo para balance y velocidad (siguiendo el patrón del notebook).
    rng = np.random.default_rng(42)
    if len(ref) > n_per_class:
        ref = ref[rng.choice(len(ref), n_per_class, replace=False)]
    if len(prod) > n_per_class:
        prod = prod[rng.choice(len(prod), n_per_class, replace=False)]

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    detector = ClassifierDrift(
        ref,
        model=clf,
        backend="sklearn",
        p_val=p_val,
        train_size=0.75,
    )
    res = detector.predict(prod)

    # Reentrenamos un RF sobre todo el ref+prod para extraer feature importances
    # explicativas (alibi-detect no las expone directamente).
    X = np.vstack([ref, prod])
    y = np.array([0] * len(ref) + [1] * len(prod))
    clf_imps = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42)
    clf_imps.fit(X, y)
    importances = {f: float(i) for f, i in zip(features, clf_imps.feature_importances_)}

    return {
        "is_drift": bool(res["data"]["is_drift"]),
        "auc": float(res["data"]["distance"]),
        "p_value": float(res["data"]["p_val"]),
        "importances": importances,
    }
