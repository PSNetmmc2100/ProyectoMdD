import pandas as pd
from tabulate import tabulate
from typing import Tuple, List

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='tabla1'))

import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
