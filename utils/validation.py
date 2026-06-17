"""
utils/validation.py
Protocol de Nested Cross-Validation compartit pels tres notebooks.

Cada notebook crida `nested_cv` amb el seu estimador i obté:
  - probabilitats OOF per a cada infant (per construir el model híbrid)
  - mètriques per fold i per llavor

La separació hold-out es fa a `data_loading.py`; aquí es treballa
únicament amb la partició d'entrenament.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from utils.metrics import compute_metrics, aggregate_metrics


def nested_cv(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    param_grid: dict | list[dict],
    feature_names: list[str],
    outer_splits: int = 5,
    inner_splits: int = 3,
    seeds: list[int] = (0, 1, 2),
    scoring: str = "f1",
    search_type: str = "grid",   # "grid" | "random"
    n_iter: int = 30,            # només per a RandomizedSearchCV
    threshold: float = 0.5,
) -> dict:
    """
    Nested Cross-Validation estratificada.

    Paràmetres
    ----------
    estimator    : Pipeline o estimador sklearn/imbalanced-learn compatible
    X, y         : features i etiquetes (conjunt d'entrenament, sense hold-out)
    groups       : nom de l'infant per a cada mostra (no s'usa aquí per a
                   agrupació, però es propaga als OOF perquè el notebook 3
                   pugui fer el merge sense ambigüitats)
    param_grid   : espai d'hiperparàmetres per a GridSearchCV / RandomizedSearchCV
    feature_names: noms de les columnes de X (per a SHAP / importàncies)
    outer_splits : nombre de folds externs (default 5)
    inner_splits : nombre de folds interns (default 3)
    seeds        : llavors per repetir el bucle extern (default [0, 1, 2])
    scoring      : mètrica per a la cerca d'hiperparàmetres (default 'f1')
    search_type  : 'grid' per GridSearchCV, 'random' per RandomizedSearchCV
    n_iter       : iteracions de RandomizedSearchCV
    threshold    : llindar de decisió per a les mètriques (default 0.5)

    Retorna
    -------
    dict amb:
        oof_records  : llista de dicts {nom, prob_dislexia, fold, seed, y_true}
        fold_metrics : llista de dicts de mètriques per (seed, fold)
        agg_metrics  : mètriques agregades (mean ± std sobre tots els folds i llavors)
        best_params  : llista de dicts {seed, fold, params}
    """
    oof_records: list[dict] = []
    fold_metrics: list[dict] = []
    best_params_list: list[dict] = []

    for seed in seeds:
        outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            groups_val = groups[val_idx]

            # Cerca d'hiperparàmetres sobre el fold intern
            if search_type == "random":
                search = RandomizedSearchCV(
                    estimator,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=inner_cv,
                    scoring=scoring,
                    n_jobs=-1,
                    random_state=seed,
                    refit=True,
                )
            else:
                search = GridSearchCV(
                    estimator,
                    param_grid=param_grid,
                    cv=inner_cv,
                    scoring=scoring,
                    n_jobs=-1,
                    refit=True,
                )

            search.fit(X_tr, y_tr)
            best_est = search.best_estimator_

            # Probabilitats OOF
            y_prob = best_est.predict_proba(X_val)[:, 1]

            # Guardar OOF per infant
            for name, prob, y_true in zip(groups_val, y_prob, y_val):
                oof_records.append({
                    "nom": name,
                    "prob_dislexia": prob,
                    "fold": fold_idx,
                    "seed": seed,
                    "y_true": int(y_true),
                })

            # Mètriques del fold
            m = compute_metrics(y_val, y_prob, threshold)
            m["seed"] = seed
            m["fold"] = fold_idx
            fold_metrics.append(m)

            best_params_list.append({
                "seed": seed,
                "fold": fold_idx,
                "params": search.best_params_,
            })

    # Mètriques numèriques (sense seed/fold) per agregar
    numeric_metrics = [
        {k: v for k, v in m.items() if k not in ("seed", "fold")}
        for m in fold_metrics
    ]
    agg = aggregate_metrics(numeric_metrics)

    return {
        "oof_records": oof_records,
        "fold_metrics": fold_metrics,
        "agg_metrics": agg,
        "best_params": best_params_list,
    }


def oof_to_dataframe(oof_records: list[dict]) -> pd.DataFrame:
    """Converteix la llista OOF a DataFrame i agrega per infant (mitjana)."""
    df = pd.DataFrame(oof_records)
    # Probabilitat OOF agregada per infant (mitjana sobre folds i llavors)
    agg = (
        df.groupby("nom")
        .agg(prob_dislexia=("prob_dislexia", "mean"), y_true=("y_true", "first"))
        .reset_index()
    )
    return df, agg
