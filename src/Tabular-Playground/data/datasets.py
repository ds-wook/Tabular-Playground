import pandas as pd

path = "../../input/tabular-playground-series-jan-2021/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
submission = pd.read_csv(path + "sample_submission.csv")

X = train.drop(["id", "target"], axis=1)
y = train["target"]

X_test = test.drop(["id"], axis=1)
