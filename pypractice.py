# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pandas import DataFrame, Series


x = np.linspace(0, 20, 100)
plt.plot(x, np.sin(x))
plt.show()


# %%
df = DataFrame(
    {'V': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]})
df['Q'] = df.expanding(4).quantile(0.9)
df
