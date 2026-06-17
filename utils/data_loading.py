"""
utils/data_loading.py
Càrrega i preparació del dataset tabular per als tres notebooks.
El hold-out del 20 % (llavor 42) es genera aquí i és idèntic per a tots.

CONTRAT DE DIVISIÓ:
  - El notebook 06 crida load_dataset(split_path=...) per primera vegada → genera
    la divisió amb random_state=42 i la desa a split_path com a índexs CSV.
  - Els notebooks 07 i 08 criden load_dataset(split_path=...) → carreguen
    exactament els mateixos índexs, independentment de la versió de scikit-learn
    o de possibles reordenacions del CSV.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_PATH = Path(__file__).parents[1] / "datasets" / "dataset.csv"

# Llengües majoritàries que reben la seva pròpia columna one-hot
MAJOR_LANGS = {"català", "castellà"}


def load_raw() -> pd.DataFrame:
    """Retorna el CSV sencer sense cap transformació."""
    return pd.read_csv(DATASET_PATH)


def encode_llengua(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encoding de llengua_materna amb categoria 'altres'."""
    df = df.copy()
    df["llengua_norm"] = df["llengua_materna"].apply(
        lambda x: x if x in MAJOR_LANGS else "altres"
    )
    dummies = pd.get_dummies(df["llengua_norm"], prefix="llengua", drop_first=False)
    # Assegura que les tres columnes existeixin sempre
    for col in ["llengua_català", "llengua_castellà", "llengua_altres"]:
        if col not in dummies.columns:
            dummies[col] = 0
    df = pd.concat([df.drop(columns=["llengua_materna", "llengua_norm"]), dummies], axis=1)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Retorna les columnes que s'usaran com a features (X)."""
    drop = {"nom", "escola", "classe", "dislexia"}
    return [c for c in df.columns if c not in drop]


def load_dataset(random_state: int = 42, split_path: Path | None = None) -> dict:
    """
    Càrrega completa: codificació, divisió hold-out 20 %.

    Parameters
    ----------
    random_state : int
        Llavor per a la divisió (ignorada si split_path existeix).
    split_path : Path | None
        Ruta al fitxer JSON que desa/carrega els índexs de la divisió.
        - Si el fitxer existeix → carrega els índexs (garanteix la mateixa
          divisió a tots els notebooks, sigui quina sigui la versió de sklearn).
        - Si el fitxer NO existeix → genera la divisió amb random_state i
          la desa per a reutilitzacions posteriors.
        - Si és None → genera la divisió sense desar-la (ús intern).

    Returns
    -------
    dict amb claus:
        X_train, y_train, groups_train   → per a nested CV
        X_hold, y_hold, groups_hold      → hold-out final (no tocar fins al final)
        feature_names                    → llista de noms de features
        df_train, df_hold                → DataFrames complets per a diagnòstics
        split_source                     → "loaded" | "generated" (per a logging)
    """
    df = load_raw()
    df = encode_llengua(df)

    y = df["dislexia"].astype(int).values
    groups = df["nom"].values  # identificador únic per infant

    feat_cols = get_feature_columns(df)
    X = df[feat_cols].values

    idx = np.arange(len(df))

    split_source: str
    if split_path is not None and Path(split_path).exists():
        with open(split_path) as f:
            saved = json.load(f)
        idx_train = np.array(saved["idx_train"])
        idx_hold = np.array(saved["idx_hold"])
        split_source = "loaded"
    else:
        idx_train, idx_hold = train_test_split(
            idx,
            test_size=0.20,
            stratify=y,
            random_state=random_state,
        )
        split_source = "generated"
        if split_path is not None:
            Path(split_path).parent.mkdir(parents=True, exist_ok=True)
            with open(split_path, "w") as f:
                json.dump({
                    "idx_train": idx_train.tolist(),
                    "idx_hold": idx_hold.tolist(),
                    "random_state": random_state,
                    "n_total": int(len(df)),
                }, f, indent=2)

    return {
        "X_train": X[idx_train],
        "y_train": y[idx_train],
        "groups_train": groups[idx_train],
        "X_hold": X[idx_hold],
        "y_hold": y[idx_hold],
        "groups_hold": groups[idx_hold],
        "feature_names": feat_cols,
        "df_train": df.iloc[idx_train].reset_index(drop=True),
        "df_hold": df.iloc[idx_hold].reset_index(drop=True),
        "split_source": split_source,
    }
