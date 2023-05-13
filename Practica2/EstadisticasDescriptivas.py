import pandas as pd

# carga los datos de un archivo csv
df = pd.read_csv('Genshin charac rev (by banner).csv')

# calcula la media, mediana y desviación estándar de todas las columnas numéricas
print(df.describe())

# calcula la correlación entre todas las columnas numéricas
print(df.corr())

# calcula la frecuencia de cada valor en una columna
print(df['banner_days'].value_counts())