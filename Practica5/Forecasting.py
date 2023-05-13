import pandas as pd
import statsmodels.api as sm

# cargar los datos desde un archivo csv
df = pd.read_csv('Genshin charac rev (by banner).csv', index_col='start_date', parse_dates=True)

# ajustar el modelo Holt-Winters
model = sm.tsa.ExponentialSmoothing(df['avg_revenue'], seasonal_periods=12, trend='add', seasonal='add').fit()

# hacer pronósticos utilizando el modelo
predicciones = model.predict(start='star_date', end='end_date')

# imprimir las predicciones
print(predicciones)
