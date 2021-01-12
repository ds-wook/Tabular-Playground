# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
# %%
path = "../input/tabular-playground-series-jan-2021/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
train.head()
# %%
min_max = MinMaxScaler()

for col in train.columns[1:-1]:
    train[col] = min_max.fit_transform(train[col].values.reshape(-1, 1))

train.head()
# %%
for col in train.columns[1:-1]:
    f, ax = plt.subplots(figsize=(20, 8))
    sns.boxplot(train[col], ax=ax)
    plt.show()
# %%
