
import os
import json
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features_fix import fix_columns, fit_encoder, apply_encoder



print("Chargement des donnees...")

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").squeeze("columns")
y_test = pd.read_csv("y_test.csv").squeeze("columns")
# Correction du prix : division par 10
y_train = y_train / 10
y_test = y_test / 10

print("Prix corriges par division par 10.")

print("Donnees chargees avec succes !")

PRIX_MIN = 100_000
PRIX_MAX = 8_000_000

mask_train = y_train.between(PRIX_MIN, PRIX_MAX)
mask_test = y_test.between(PRIX_MIN, PRIX_MAX)
X_train, y_train = X_train[mask_train], y_train[mask_train]
X_test, y_test = X_test[mask_test], y_test[mask_test]
print("Lignes gardées :", len(X_train), len(X_test))


print("Correction des colonnes...")

X_train = fix_columns(X_train)
X_test = fix_columns(X_test)



print("Encodage des variables...")

encoder = fit_encoder(X_train)

X_train = apply_encoder(X_train, encoder)
X_test = apply_encoder(X_test, encoder)



print("Verification des donnees...")

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Les colonnes Train et Test ne correspondent pas !")

if X_train.select_dtypes(include="object").shape[1] > 0:
    raise ValueError("Il reste des colonnes textuelles !")

print("Nombre de features :", X_train.shape[1])
print("Taille Train :", X_train.shape)
print("Taille Test :", X_test.shape)

print("Creation du modele XGBoost...")

model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror",
    n_jobs=-1
)


print("Entrainement du modele...")

model.fit(X_train, y_train)

print("Entrainement termine !")




print("Prediction des prix...")

y_pred = model.predict(X_test)



mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("\n========== RESULTATS XGBOOST ==========")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")


os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

joblib.dump(model, "model_xgb.pkl")


results = {
    "model": "XGBoost",
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2)
}

with open("results_xgb.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nModele et resultats sauvegardes avec succes !")
