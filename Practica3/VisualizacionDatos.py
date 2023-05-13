import matplotlib.pyplot as plt
import pandas as pd

# cargar los datos desde un archivo csv
df = pd.read_csv('openpowerlifting.csv')

# crear un histograma de una columna
plt.hist(df['TotalKg'])
plt.title('Total Peso')
plt.xlabel('Mas de 100')
plt.ylabel('Menos de 100')
plt.show()

# crear un diagrama de dispersión de dos columnas
plt.scatter(df['Age'], df['BodyweightKg'])
plt.title('Levantamientos por Edad')
plt.xlabel('Edad')
plt.ylabel('Pesas en Kg')
plt.show()