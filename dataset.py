import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

df = pd.read_csv("data/log.csv")

n, bins, patches = plt.hist(df['angle'], 10)
plt.hist(df['angle'], bins=10)
plt.show()
