import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# cargar los datos desde un archivo csv
df = pd.read_csv('Genshin charac rev (by banner).csv')

# seleccionar las características y la variable objetivo
X = df.drop(['5_star_characters'], axis=1)
y = df['avg_revenue']

# dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ajustar el modelo de bosques aleatorios
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# hacer predicciones utilizando el modelo
predicciones = model.predict(X_test)

# evaluar la precisión del modelo
precision = accuracy_score(y_test, predicciones)
print("Precisión: ", precision)
