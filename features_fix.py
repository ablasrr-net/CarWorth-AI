import ast
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def count_equipment(s):
    try:
        return len(ast.literal_eval(s))
    except Exception:
        return 0


def fix_columns(df):
    """Convertit en nombres les colonnes restées en texte."""
    df = df.copy()
    # "9 CV" -> 9
    df["Fiscal Power"] = (
        df["Fiscal Power"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    )
    # Liste d'équipements -> nombre d'équipements
    df["Equipment"] = df["Equipment"].apply(count_equipment)
    # Yes/No -> 1/0 (manquant -> -1)
    df["First Owner"] = df["First Owner"].map({"Yes": 1, "No": 0}).fillna(-1)
    # Portes manquantes -> 4
    df["Number of Doors"] = df["Number of Doors"].fillna(4)
    return df


CAT_COLS = ["Location", "Sector"]


def fit_encoder(X_train):
    """Apprend l'encodage de Location et Sector sur le train uniquement."""
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(X_train[CAT_COLS].astype(str))
    return enc


def apply_encoder(df, enc):
    df = df.copy()
    df[CAT_COLS] = enc.transform(df[CAT_COLS].astype(str))
    return df