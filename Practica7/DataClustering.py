import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# cargar los datos desde un archivo csv
df = pd.read_csv('Genshin charac rev (by banner).csv')

# normalizar los datos
scaler = StandardScaler()
df_norm = scaler.fit_transform(df)

# reducir la dimensionalidad de los datos utilizando PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_norm)

# ajustar el modelo de K-means
model = KMeans(n_clusters=3, random_state=42)
model.fit(df_norm)

# hacer predicciones utilizando el modelo
predicciones = model.predict(df_norm)

# visualizar los resultados del clustering
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=predicciones)
plt.xlabel('revenue')
plt.ylabel('avg_revenue')
plt.show()
