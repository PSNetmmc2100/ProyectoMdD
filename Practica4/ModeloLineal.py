import pandas as pd
from sklearn.linear_model import LinearRegression

# cargar los datos desde un archivo csv
df = pd.read_csv('Genshin charac rev (by banner).csv')

# seleccionar las características y la variable objetivo
X = df[['Revenue', 'avg_revenue', '5_star_characters']]
y = df['banner_days']

# ajustar el modelo de regresión lineal
model = LinearRegression().fit(X, y)

# imprimir los coeficientes del modelo
print(model.coef_)

# hacer predicciones utilizando el modelo
predicciones = model.predict(X)
