import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features_fix import fix_columns, fit_encoder, apply_encoder

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

y_train = y_train / 10
y_test = y_test / 10

print("Prix corriges par division par 10.")

PRIX_MIN = 100_000
PRIX_MAX = 8_000_000

mask_train = y_train.between(PRIX_MIN, PRIX_MAX)
mask_test = y_test.between(PRIX_MIN, PRIX_MAX)
X_train, y_train = X_train[mask_train], y_train[mask_train]
X_test, y_test = X_test[mask_test], y_test[mask_test]
print("Lignes gardées :", len(X_train), len(X_test))

X_train = fix_columns(X_train)
X_test = fix_columns(X_test)

enc = fit_encoder(X_train)
X_train = apply_encoder(X_train, enc)
X_test = apply_encoder(X_test, enc)

print("Types des colonnes (aucun 'object' attendu) :")
print(X_train.dtypes)


model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)


preds = model.predict(X_test)
results = {
    "model": "RandomForest",
    "MAE": float(mean_absolute_error(y_test, preds)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
    "R2": float(r2_score(y_test, preds)),
}
print("\nRésultats :", results)

#Importance des variables
importances = pd.Series(model.feature_importances_, index=X_train.columns)
print("\nVariables les plus importantes :")
print(importances.sort_values(ascending=False).head(10))


joblib.dump(model, "model_rf.pkl")
joblib.dump(enc, "encoder_location_sector.pkl")
with open("results_rf.json", "w") as f:
    json.dump(results, f)

print("\nFichiers sauvegardés : model_rf.pkl, encoder_location_sector.pkl,results_rf.json")