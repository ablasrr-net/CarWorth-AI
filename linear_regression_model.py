from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns

# avant de charger le modèle, on charge les données nettoyées       
df = pd.read_csv('data_cleaned.csv')
#avant d'entrainer le modele, on valide les dimensions du x_train et y_train
X_train=pd.read_csv('X_train.csv')
y_train=pd.read_csv('y_train.csv')
X_test=pd.read_csv("X_test.csv")
y_test=pd.read_csv("y_test.csv")
print("X_train :", X_train.shape)
print("y_train :", y_train.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)
# creer une instance du modèle de régression linéaire
model=LinearRegression()
model.fit(X_train,y_train)
#Maintenant que le modèle est entraîné, utilisons X_test.
y_pred = model.predict(X_test)
#visualisation des résultats
# CSV de cible lu comme DataFrame (n, 1) : convertir en vecteurs 1D pour pandas.
y_test_1d = y_test.squeeze()
y_pred_1d = np.asarray(y_pred).ravel()
resultats = pd.DataFrame({
    "Prix réel": y_test_1d,
    "Prix prédit": y_pred_1d,
})

print(resultats.head(10))
#evaluation du modèle
mse = mean_squared_error(y_test_1d, y_pred_1d)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_1d, y_pred_1d)
r2 = r2_score(y_test_1d, y_pred_1d)
print("Erreur quadratique moyenne (MSE):", mse)
print("Racine de l'erreur quadratique moyenne (RMSE):", rmse)
print("Erreur absolue moyenne (MAE):", mae)
print("Coefficient de détermination (R²):", r2)
#construction d'un graphique pour visualiser les résultats
plt.figure(figsize=(8, 6))

plt.scatter(y_test_1d, y_pred_1d, alpha=0.4)

min_val = min(y_test_1d.min(), y_pred_1d.min())
max_val = max(y_test_1d.max(), y_pred_1d.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    'r--',
    label="Prédiction parfaite"
)

plt.xlabel("Prix réel (MAD)")
plt.ylabel("Prix prédit (MAD)")
plt.title("Linear Regression : prix réels vs prix prédits")

plt.legend()
plt.grid(True)
plt.show()
