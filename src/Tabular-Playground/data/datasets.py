from typing import Tuple
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

path = "../../input/tabular-playground-series-jan-2021/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
submission = pd.read_csv(path + "sample_submission.csv")

X = train.drop(["id", "target"], axis=1)
y = train["target"]

X_test = test.drop(["id"], axis=1)


# def normalize_transformer(
#     train: pd.DataFrame, test: pd.DataFrame
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     quantile = QuantileTransformer(output_distribution="normal")
#     normal_train = quantile.fit_transform(train)
#     normal_test = quantile.fit_transform(test)
#     normal_train = pd.DataFrame(normal_train, columns=train.columns)
#     normal_test = pd.DataFrame(normal_test, columns=test.columns)
#     return normal_train, normal_test


# X, X_test = normalize_transformer(X, X_test)
