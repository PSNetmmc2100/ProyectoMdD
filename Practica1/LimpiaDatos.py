import pandas as pd

# cargar los datos desde un archivo csv
df = pd.read_csv('Spotify_final_dataset Actualizada.csv')

# eliminar filas duplicadas
df = df.drop_duplicates()

# eliminar filas con valores faltantes
df = df.dropna()

# convertir todas las cadenas de texto a minúsculas
df = df.apply(lambda x: x.str.lower() if x.dtype == "object" else x)

# eliminar espacios en blanco al principio y al final de las cadenas de texto
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# reemplazar los valores de una columna con otros valores
df['columna'] = df['columna'].replace({'valor_a_reemplazar': 'nuevo_valor'})

# eliminar caracteres no alfanuméricos de una columna
df['columna'] = df['columna'].replace('[^\w\s]','',regex=True)

# guardar los datos limpios en un nuevo archivo csv
df.to_csv('Spotify_final_dataset Limpia.csv', index=False)