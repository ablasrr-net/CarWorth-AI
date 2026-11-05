import numpy as np 
import pandas as pd

print("Chargement des donnees...")
df = pd.read_csv("cars_dataframe.csv")

print(f"Nombre de lignes au départ : {len(df)}")

print("Information sur le dataset:")
print(df.info())

colonnes_importantes = ['Price', 'Year', 'Mileage', 'Fuel', 'Gearbox', 'Brand', 'Model', 'Condition', 'Origin']
df_cleaned = df.dropna(subset=colonnes_importantes).copy()

print(f"Lignes supprimees car valeurs manquantes : {len(df) - len(df_cleaned)}")

print("Convertion des colonnes 'Price' et 'Mileage' en numerique...")
df_cleaned['Price'] = pd.to_numeric(df_cleaned['Price'].astype(str).str.replace(r'\D', '', regex=True),errors='coerce')
df_cleaned['Mileage'] = pd.to_numeric(
    df_cleaned['Mileage'].astype(str).str.replace(r'\D', '', regex=True), 
    errors='coerce'
)

df_cleaned['Year'] = pd.to_numeric(df_cleaned['Year'], errors='coerce')

df_cleaned = df_cleaned[
    (df_cleaned['Price'] > 0) & 
    (df_cleaned['Mileage'] >= 0) & 
    (df_cleaned['Year'] > 1990)
].copy()


print(f"Lignes supprimees car valeurs aberrantes (prix/km impossibles) : {len(df) - len(df_cleaned)}")


print(f"\nNombre de lignes finales et propres : {len(df_cleaned)}")


df_cleaned.to_csv('data_cleaned.csv',index=False)

