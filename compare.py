import json, shutil, pandas as pd

rf = json.load(open("results_rf.json"))
xgb = json.load(open("results_xgb.json"))

table = pd.DataFrame([rf, xgb]).set_index("model")
print(table.round(3))

best = table["RMSE"].idxmin()
print(f"\nMeilleur modèle : {best}")

src = "model_rf.pkl" if best == "RandomForest" else "model_xgb.pkl"
shutil.copy(src, "model.pkl") 