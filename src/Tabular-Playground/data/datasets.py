from typing import Tuple
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.mixture import GaussianMixture


def normalize_transformer(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    quantile = QuantileTransformer(output_distribution="normal")
    normal_train = quantile.fit_transform(train)
    normal_test = quantile.fit_transform(test)
    normal_train = pd.DataFrame(normal_train, columns=train.columns)
    normal_test = pd.DataFrame(normal_test, columns=test.columns)
    return normal_train, normal_test


def get_gmm_class_feature(
    feat: str, n: int, X: pd.DataFrame, X_test: pd.DataFrame
) -> None:
    gmm = GaussianMixture(n_components=n, random_state=42)

    gmm.fit(X[feat].values.reshape(-1, 1))

    X[f"{feat}_class"] = gmm.predict(X[feat].values.reshape(-1, 1))
    X_test[f"{feat}_class"] = gmm.predict(X_test[feat].values.reshape(-1, 1))
    return X, X_test


path = "../../input/tabular-playground-series-jan-2021/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
submission = pd.read_csv(path + "sample_submission.csv")

X = train.drop(["id", "target"], axis=1)
y = train["target"]

X_test = test.drop(["id"], axis=1)

# cols = X.columns
# nums = [4, 10, 6, 4, 3, 2, 3, 4, 4, 8, 5, 4, 6, 6]

# for c, n in zip(cols, nums):
#     X, X_test = get_gmm_class_feature(c, n, X, X_test)
