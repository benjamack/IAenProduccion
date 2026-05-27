from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def compute_ks_drift(
    X_ref: pd.DataFrame,
    X_prod: pd.DataFrame,
    features: list[str],
    p_val: float = 0.05,
) -> dict:
    ref = X_ref[features].astype(np.float32).values
    prod = X_prod[features].astype(np.float32).values

    # Bonferroni: el umbral por feature se corrige por la cantidad de tests.
    p_val_corrected = p_val / max(len(features), 1)

    per_feature = []
    n_drifted = 0
    p_values: list[float] = []
    for i, feat in enumerate(features):
        d, p = ks_2samp(ref[:, i], prod[:, i])
        is_feat_drift = bool(p < p_val_corrected)
        n_drifted += int(is_feat_drift)
        p_values.append(float(p))
        per_feature.append({
            "feature": feat,
            "D_KS": float(d),
            "p_value": float(p),
            "drift": is_feat_drift,
        })

    return {
        "is_drift": any(pf["drift"] for pf in per_feature),
        "per_feature": per_feature,
        "n_drifted": n_drifted,
        "min_p_value": float(min(p_values)) if p_values else 1.0,
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

    # Submuestreo para balance y velocidad (igual que con alibi-detect).
    rng = np.random.default_rng(42)
    if len(ref) > n_per_class:
        ref = ref[rng.choice(len(ref), n_per_class, replace=False)]
    if len(prod) > n_per_class:
        prod = prod[rng.choice(len(prod), n_per_class, replace=False)]

    X = np.vstack([ref, prod])
    y = np.array([0] * len(ref) + [1] * len(prod))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=42, stratify=y,
    )

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, 1]

    auc = float(roc_auc_score(y_test, y_score))

    # p-value via Mann-Whitney U: bajo H0 (ref ≡ prod) los scores de ambas clases
    # son intercambiables. AUC = U / (n_pos * n_neg), así que Mann-Whitney da el
    # p-value exacto del test de dos muestras sobre los scores predichos.
    scores_pos = y_score[y_test == 1]
    scores_neg = y_score[y_test == 0]
    _u, p_value = mannwhitneyu(scores_pos, scores_neg, alternative="greater")

    # Segundo RF sobre todo el dataset para feature importances explicativas.
    clf_imps = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42)
    clf_imps.fit(X, y)
    importances = {f: float(i) for f, i in zip(features, clf_imps.feature_importances_)}

    return {
        "is_drift": bool(p_value < p_val),
        "auc": auc,
        "p_value": float(p_value),
        "importances": importances,
    }
