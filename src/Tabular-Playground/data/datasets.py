import pandas as pd
from sklearn.model_selection import train_test_split

path = "../../input/tabular-playground-series-jan-2021/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
submission = pd.read_csv(path + 'sample_submission.csv')

X = train.drop(["id", "target"], axis=1)
y = train["target"]
X_test = test.drop(["id"], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
