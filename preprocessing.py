import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


print("Chargement des donnees netytoyees...")
df = pd.read_csv('data_cleaned.csv')
print(f"Nombre de lignes chargees : {len(df)}")

#plus facile de travailler avec l'age que l'annee de fabrication
df['Age'] = 2024 - df['Year']
print ("colenne 'Age' creee avec succes.")

#transform les colonnes texte en numerique pour le traitement le model ne connait pas les str 
colonnes_a_encoder = ['Brand', 'Model', 'Gearbox', 'Fuel', 'Condition', 'Origin']


print("Encodage des colonnes texte en cours...")
for col in colonnes_a_encoder:
    if col in df.columns:
        
        encodeur = LabelEncoder()
        df[col] = encodeur.fit_transform(df[col].astype(str))



X = df.drop('Price', axis=1)
y = df['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Preparation terminee")

print("Taille des donnees de traitement : " , X_train.shape)
print("Taille des donnees de test : " , X_test.shape)

X_train.to_csv('X_train.csv' , index=False)
X_test.to_csv('X_test.csv' , index=False)
y_train.to_csv('y_train.csv' , index=False)
y_test.to_csv('y_test.csv' , index=False)
